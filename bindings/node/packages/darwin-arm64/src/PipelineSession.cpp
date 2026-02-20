#include "PipelineSession.h"
#include "PipelineModel.h"
#include "PipelinePushWorker.h"
#include "PipelineFinalizeWorker.h"
#include <cstring>

Napi::FunctionReference PipelineSession::constructor;

struct TSFNCallbackData {
    std::vector<AlignedSegment> segments;
    std::vector<float> audio;
};

static Napi::Object MarshalSegment(Napi::Env env, const AlignedSegment& seg) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("speaker", Napi::String::New(env, seg.speaker));
    obj.Set("start", Napi::Number::New(env, seg.start));
    obj.Set("duration", Napi::Number::New(env, seg.duration));
    obj.Set("text", Napi::String::New(env, seg.text));
    return obj;
}

Napi::Array MarshalSegments(Napi::Env env, const std::vector<AlignedSegment>& segments) {
    Napi::Array arr = Napi::Array::New(env, segments.size());
    for (size_t i = 0; i < segments.size(); i++) {
        arr.Set(static_cast<uint32_t>(i), MarshalSegment(env, segments[i]));
    }
    return arr;
}

Napi::Object PipelineSession::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "PipelineSession", {
        InstanceMethod<&PipelineSession::Push>("push"),
        InstanceMethod<&PipelineSession::Finalize>("finalize"),
        InstanceMethod<&PipelineSession::Close>("close"),
        InstanceAccessor<&PipelineSession::GetIsClosed>("isClosed"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("PipelineSession", func);
    return exports;
}

PipelineSession::PipelineSession(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PipelineSession>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsFunction()) {
        Napi::TypeError::New(env,
            "PipelineSession cannot be constructed directly. Use model.createSession(callback)")
            .ThrowAsJavaScriptException();
        return;
    }

    PipelineModel* model = info[0].As<Napi::External<PipelineModel>>().Data();
    Napi::Function callback = info[1].As<Napi::Function>();

    if (model->IsClosed()) {
        Napi::Error::New(env, "Model is closed").ThrowAsJavaScriptException();
        return;
    }

    PipelineConfig srcConfig = model->BuildConfig();

    seg_model_path_ = srcConfig.diarization.seg_model_path;
    emb_model_path_ = srcConfig.diarization.emb_model_path;
    plda_path_ = srcConfig.diarization.plda_path;
    coreml_path_ = srcConfig.diarization.coreml_path;
    seg_coreml_path_ = srcConfig.diarization.seg_coreml_path;
    if (srcConfig.transcriber.whisper_model_path)
        whisper_model_path_ = srcConfig.transcriber.whisper_model_path;
    if (srcConfig.vad_model_path)
        vad_model_path_ = srcConfig.vad_model_path;
    if (srcConfig.transcriber.language)
        language_ = srcConfig.transcriber.language;
    n_threads_ = srcConfig.transcriber.n_threads;

    use_gpu_ = srcConfig.transcriber.use_gpu;
    flash_attn_ = srcConfig.transcriber.flash_attn;
    gpu_device_ = srcConfig.transcriber.gpu_device;
    use_coreml_ = srcConfig.transcriber.use_coreml;
    no_prints_ = srcConfig.transcriber.no_prints;

    translate_ = srcConfig.transcriber.translate;
    detect_language_ = srcConfig.transcriber.detect_language;
    temperature_ = srcConfig.transcriber.temperature;
    temperature_inc_ = srcConfig.transcriber.temperature_inc;
    no_fallback_ = srcConfig.transcriber.no_fallback;
    beam_size_ = srcConfig.transcriber.beam_size;
    best_of_ = srcConfig.transcriber.best_of;
    entropy_thold_ = srcConfig.transcriber.entropy_thold;
    logprob_thold_ = srcConfig.transcriber.logprob_thold;
    no_speech_thold_ = srcConfig.transcriber.no_speech_thold;
    if (srcConfig.transcriber.prompt) prompt_ = srcConfig.transcriber.prompt;
    no_context_ = srcConfig.transcriber.no_context;
    suppress_blank_ = srcConfig.transcriber.suppress_blank;
    suppress_nst_ = srcConfig.transcriber.suppress_nst;

    js_callback_ = Napi::Persistent(callback);

    tsfn_ = Napi::ThreadSafeFunction::New(
        env,
        callback,
        "PipelineCallback",
        0,
        1
    );

    PipelineConfig config{};
    config.diarization.seg_model_path = seg_model_path_;
    config.diarization.emb_model_path = emb_model_path_;
    config.diarization.plda_path = plda_path_;
    config.diarization.coreml_path = coreml_path_;
    config.diarization.seg_coreml_path = seg_coreml_path_;
    config.transcriber.whisper_model_path = whisper_model_path_.empty() ? nullptr : whisper_model_path_.c_str();
    config.transcriber.n_threads = n_threads_;
    config.transcriber.language = language_.empty() ? nullptr : language_.c_str();
    config.vad_model_path = vad_model_path_.empty() ? nullptr : vad_model_path_.c_str();

    config.transcriber.use_gpu = use_gpu_;
    config.transcriber.flash_attn = flash_attn_;
    config.transcriber.gpu_device = gpu_device_;
    config.transcriber.use_coreml = use_coreml_;
    config.transcriber.no_prints = no_prints_;

    config.transcriber.translate = translate_;
    config.transcriber.detect_language = detect_language_;
    config.transcriber.temperature = temperature_;
    config.transcriber.temperature_inc = temperature_inc_;
    config.transcriber.no_fallback = no_fallback_;
    config.transcriber.beam_size = beam_size_;
    config.transcriber.best_of = best_of_;
    config.transcriber.entropy_thold = entropy_thold_;
    config.transcriber.logprob_thold = logprob_thold_;
    config.transcriber.no_speech_thold = no_speech_thold_;
    config.transcriber.prompt = prompt_.empty() ? nullptr : prompt_.c_str();
    config.transcriber.no_context = no_context_;
    config.transcriber.suppress_blank = suppress_blank_;
    config.transcriber.suppress_nst = suppress_nst_;

    state_ = pipeline_init(config, pipeline_cb, this);
    if (!state_) {
        tsfn_.Release();
        tsfn_released_ = true;
        Napi::Error::New(env, "Failed to initialize pipeline state")
            .ThrowAsJavaScriptException();
        return;
    }
}

PipelineSession::~PipelineSession() {
    Cleanup();
}

void PipelineSession::Cleanup() {
    if (state_) {
        pipeline_free(state_);
        state_ = nullptr;
    }
    if (!tsfn_released_) {
        tsfn_.Release();
        tsfn_released_ = true;
    }
    closed_ = true;
}

void PipelineSession::pipeline_cb(const std::vector<AlignedSegment>& segments,
                                   const std::vector<float>& audio,
                                   void* user_data) {
    auto* self = static_cast<PipelineSession*>(user_data);

    {
        std::lock_guard<std::mutex> lock(self->segments_mutex_);
        self->last_segments_ = segments;
    }

    auto* data = new TSFNCallbackData{segments, audio};

    auto status = self->tsfn_.NonBlockingCall(data,
        [](Napi::Env env, Napi::Function jsCallback, TSFNCallbackData* cbData) {
            Napi::Array segmentsArr = MarshalSegments(env, cbData->segments);

            Napi::ArrayBuffer audioBuf = Napi::ArrayBuffer::New(
                env, cbData->audio.size() * sizeof(float));
            if (!cbData->audio.empty()) {
                memcpy(audioBuf.Data(), cbData->audio.data(),
                       cbData->audio.size() * sizeof(float));
            }
            Napi::Float32Array audioArr = Napi::Float32Array::New(
                env, cbData->audio.size(), audioBuf, 0);

            jsCallback.Call({segmentsArr, audioArr});

            delete cbData;
        });

    if (status != napi_ok) {
        delete data;
    }
}

Napi::Value PipelineSession::Push(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (busy_) {
        Napi::Error::New(env, "Session is busy with another operation")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "Expected Float32Array argument")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::TypedArray typedArr = info[0].As<Napi::TypedArray>();
    if (typedArr.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "Expected Float32Array argument")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Float32Array float32Arr = info[0].As<Napi::Float32Array>();
    std::vector<float> samples(float32Arr.Data(),
                               float32Arr.Data() + float32Arr.ElementLength());

    busy_ = true;
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new PipelinePushWorker(env, this, state_,
                                          std::move(samples), deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value PipelineSession::Finalize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (busy_) {
        Napi::Error::New(env, "Session is busy with another operation")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    busy_ = true;
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new PipelineFinalizeWorker(env, this, state_, deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value PipelineSession::Close(const Napi::CallbackInfo& info) {
    Cleanup();
    return info.Env().Undefined();
}

Napi::Value PipelineSession::GetIsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), closed_);
}
