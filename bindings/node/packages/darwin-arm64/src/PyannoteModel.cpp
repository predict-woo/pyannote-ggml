#include "PyannoteModel.h"
#include "StreamingSession.h"
#include "DiarizeWorker.h"

Napi::FunctionReference PyannoteModel::constructor;

Napi::Object PyannoteModel::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "PyannoteModel", {
        InstanceMethod<&PyannoteModel::Diarize>("diarize"),
        InstanceMethod<&PyannoteModel::CreateStreamingSession>("createStreamingSession"),
        InstanceMethod<&PyannoteModel::Close>("close"),
        InstanceAccessor<&PyannoteModel::GetIsClosed>("isClosed"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("PyannoteModel", func);
    return exports;
}

PyannoteModel::PyannoteModel(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PyannoteModel>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env,
            "Expected config object with: segModelPath, embModelPath, pldaPath, coremlPath, segCoremlPath")
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

    config_.seg_model_path = getStringProp("segModelPath");
    if (env.IsExceptionPending()) return;
    config_.emb_model_path = getStringProp("embModelPath");
    if (env.IsExceptionPending()) return;
    config_.plda_path = getStringProp("pldaPath");
    if (env.IsExceptionPending()) return;
    config_.coreml_path = getStringProp("coremlPath");
    if (env.IsExceptionPending()) return;
    config_.seg_coreml_path = getStringProp("segCoremlPath");
    if (env.IsExceptionPending()) return;

    if (config.Has("zeroLatency") && config.Get("zeroLatency").IsBoolean()) {
        config_.zero_latency = config.Get("zeroLatency").As<Napi::Boolean>().Value();
    }
}

Napi::Value PyannoteModel::Diarize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Model is closed").ThrowAsJavaScriptException();
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

    // Copy data — JS GC may collect the buffer during async execution
    std::vector<float> audio(float32Arr.Data(), float32Arr.Data() + length);

    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new DiarizeWorker(env, config_, std::move(audio), deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value PyannoteModel::CreateStreamingSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Model is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Pass the config to StreamingSession's constructor via external
    Napi::External<StreamingConfig> configExt =
        Napi::External<StreamingConfig>::New(env, &config_);

    return StreamingSession::constructor.New({ configExt });
}

Napi::Value PyannoteModel::Close(const Napi::CallbackInfo& info) {
    closed_ = true;
    return info.Env().Undefined();
}

Napi::Value PyannoteModel::GetIsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), closed_);
}
