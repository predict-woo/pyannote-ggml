#include "PipelineModel.h"
#include "PipelineSession.h"
#include "TranscribeWorker.h"

Napi::FunctionReference PipelineModel::constructor;

Napi::Object PipelineModel::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "PipelineModel", {
        InstanceMethod<&PipelineModel::Transcribe>("transcribe"),
        InstanceMethod<&PipelineModel::CreateSession>("createSession"),
        InstanceMethod<&PipelineModel::Close>("close"),
        InstanceAccessor<&PipelineModel::GetIsClosed>("isClosed"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("PipelineModel", func);
    return exports;
}

PipelineModel::PipelineModel(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PipelineModel>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env,
            "Expected config object with: segModelPath, embModelPath, pldaPath, "
            "coremlPath, segCoremlPath, whisperModelPath")
            .ThrowAsJavaScriptException();
        return;
    }

    Napi::Object config = info[0].As<Napi::Object>();

    auto getStringProp = [&](const char* name) -> std::string {
        if (!config.Has(name) || !config.Get(name).IsString()) {
            Napi::TypeError::New(env,
                std::string("Config property '") + name + "' must be a non-empty string")
                .ThrowAsJavaScriptException();
            return "";
        }
        std::string val = config.Get(name).As<Napi::String>().Utf8Value();
        if (val.empty()) {
            Napi::TypeError::New(env,
                std::string("Config property '") + name + "' must be a non-empty string")
                .ThrowAsJavaScriptException();
            return "";
        }
        return val;
    };

    auto getOptionalStringProp = [&](const char* name) -> std::string {
        if (config.Has(name) && config.Get(name).IsString()) {
            return config.Get(name).As<Napi::String>().Utf8Value();
        }
        return "";
    };

    seg_model_path_ = getStringProp("segModelPath");
    if (env.IsExceptionPending()) return;
    emb_model_path_ = getStringProp("embModelPath");
    if (env.IsExceptionPending()) return;
    plda_path_ = getStringProp("pldaPath");
    if (env.IsExceptionPending()) return;
    coreml_path_ = getStringProp("coremlPath");
    if (env.IsExceptionPending()) return;
    seg_coreml_path_ = getStringProp("segCoremlPath");
    if (env.IsExceptionPending()) return;
    whisper_model_path_ = getStringProp("whisperModelPath");
    if (env.IsExceptionPending()) return;

    vad_model_path_ = getOptionalStringProp("vadModelPath");
    language_ = getOptionalStringProp("language");

    auto getBoolProp = [&](const char* name, bool defaultVal) -> bool {
        if (config.Has(name) && config.Get(name).IsBoolean()) {
            return config.Get(name).As<Napi::Boolean>().Value();
        }
        return defaultVal;
    };

    auto getFloatProp = [&](const char* name, float defaultVal) -> float {
        if (config.Has(name) && config.Get(name).IsNumber()) {
            return static_cast<float>(config.Get(name).As<Napi::Number>().DoubleValue());
        }
        return defaultVal;
    };

    auto getIntProp = [&](const char* name, int defaultVal) -> int {
        if (config.Has(name) && config.Get(name).IsNumber()) {
            return config.Get(name).As<Napi::Number>().Int32Value();
        }
        return defaultVal;
    };

    n_threads_ = getIntProp("nThreads", 4);

    // Context options
    use_gpu_ = getBoolProp("useGpu", true);
    flash_attn_ = getBoolProp("flashAttn", true);
    gpu_device_ = getIntProp("gpuDevice", 0);
    use_coreml_ = getBoolProp("useCoreml", false);
    no_prints_ = getBoolProp("noPrints", false);

    // Decode options
    translate_ = getBoolProp("translate", false);
    detect_language_ = getBoolProp("detectLanguage", false);
    temperature_ = getFloatProp("temperature", 0.0f);
    temperature_inc_ = getFloatProp("temperatureInc", 0.2f);
    no_fallback_ = getBoolProp("noFallback", false);
    beam_size_ = getIntProp("beamSize", -1);
    best_of_ = getIntProp("bestOf", 5);
    entropy_thold_ = getFloatProp("entropyThold", 2.4f);
    logprob_thold_ = getFloatProp("logprobThold", -1.0f);
    no_speech_thold_ = getFloatProp("noSpeechThold", 0.6f);
    prompt_ = getOptionalStringProp("prompt");
    no_context_ = getBoolProp("noContext", true);
    suppress_blank_ = getBoolProp("suppressBlank", true);
    suppress_nst_ = getBoolProp("suppressNst", false);
}

PipelineModel::~PipelineModel() {
    if (!closed_) {
        closed_ = true;
    }
}

PipelineConfig PipelineModel::BuildConfig() const {
    PipelineConfig config{};
    config.diarization.seg_model_path = seg_model_path_;
    config.diarization.emb_model_path = emb_model_path_;
    config.diarization.plda_path = plda_path_;
    config.diarization.coreml_path = coreml_path_;
    config.diarization.seg_coreml_path = seg_coreml_path_;
    config.transcriber.whisper_model_path = whisper_model_path_.c_str();
    config.transcriber.n_threads = n_threads_;
    config.transcriber.language = language_.empty() ? nullptr : language_.c_str();

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

    config.vad_model_path = vad_model_path_.empty() ? nullptr : vad_model_path_.c_str();
    return config;
}

Napi::Value PipelineModel::Transcribe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Model is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (busy_) {
        Napi::Error::New(env, "Model is busy with another operation")
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
    size_t length = float32Arr.ElementLength();

    if (length == 0) {
        Napi::TypeError::New(env, "Float32Array must not be empty")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<float> audio(float32Arr.Data(), float32Arr.Data() + length);

    busy_ = true;
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new TranscribeWorker(env, this, std::move(audio), deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value PipelineModel::CreateSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Model is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 2 || !info[0].IsFunction() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Expected two callback function arguments (segments, audio)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::External<PipelineModel> modelExt =
        Napi::External<PipelineModel>::New(env, this);

    return PipelineSession::constructor.New({ modelExt, info[0], info[1] });
}

Napi::Value PipelineModel::Close(const Napi::CallbackInfo& info) {
    closed_ = true;
    return info.Env().Undefined();
}

Napi::Value PipelineModel::GetIsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), closed_);
}
