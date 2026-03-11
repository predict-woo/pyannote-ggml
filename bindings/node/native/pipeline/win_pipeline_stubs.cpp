#include <napi.h>

namespace {

constexpr const char* kUnsupportedPlatformMessage =
    "Pipeline is only supported on macOS Apple Silicon. "
    "Low-level whisper/VAD APIs remain available on this platform.";

Napi::Value RejectUnsupportedPromise(Napi::Env env) {
    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    deferred.Reject(Napi::Error::New(env, kUnsupportedPlatformMessage).Value());
    return deferred.Promise();
}

class PipelineModel : public Napi::ObjectWrap<PipelineModel> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::FunctionReference constructor;

    explicit PipelineModel(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<PipelineModel>(info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
    }

private:
    Napi::Value Transcribe(const Napi::CallbackInfo& info) {
        return RejectUnsupportedPromise(info.Env());
    }

    Napi::Value TranscribeOffline(const Napi::CallbackInfo& info) {
        return RejectUnsupportedPromise(info.Env());
    }

    Napi::Value CreateSession(const Napi::CallbackInfo& info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    Napi::Value SetLanguage(const Napi::CallbackInfo& info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    Napi::Value SetDecodeOptions(const Napi::CallbackInfo& info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    Napi::Value Close(const Napi::CallbackInfo& info) {
        return info.Env().Undefined();
    }

    Napi::Value LoadModels(const Napi::CallbackInfo& info) {
        return RejectUnsupportedPromise(info.Env());
    }

    Napi::Value SwitchWhisperMode(const Napi::CallbackInfo& info) {
        return RejectUnsupportedPromise(info.Env());
    }

    Napi::Value GetIsClosed(const Napi::CallbackInfo& info) {
        return Napi::Boolean::New(info.Env(), false);
    }

    Napi::Value GetIsLoaded(const Napi::CallbackInfo& info) {
        return Napi::Boolean::New(info.Env(), false);
    }
};

Napi::FunctionReference PipelineModel::constructor;

Napi::Object PipelineModel::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "PipelineModel", {
        InstanceMethod<&PipelineModel::Transcribe>("transcribe"),
        InstanceMethod<&PipelineModel::TranscribeOffline>("transcribeOffline"),
        InstanceMethod<&PipelineModel::CreateSession>("createSession"),
        InstanceMethod<&PipelineModel::SetLanguage>("setLanguage"),
        InstanceMethod<&PipelineModel::SetDecodeOptions>("setDecodeOptions"),
        InstanceMethod<&PipelineModel::Close>("close"),
        InstanceMethod<&PipelineModel::LoadModels>("loadModels"),
        InstanceMethod<&PipelineModel::SwitchWhisperMode>("switchWhisperMode"),
        InstanceAccessor<&PipelineModel::GetIsClosed>("isClosed"),
        InstanceAccessor<&PipelineModel::GetIsLoaded>("isLoaded"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("PipelineModel", func);
    return exports;
}

class PipelineSession : public Napi::ObjectWrap<PipelineSession> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::FunctionReference constructor;

    explicit PipelineSession(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<PipelineSession>(info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
    }

private:
    Napi::Value Push(const Napi::CallbackInfo& info) {
        return RejectUnsupportedPromise(info.Env());
    }

    Napi::Value Finalize(const Napi::CallbackInfo& info) {
        return RejectUnsupportedPromise(info.Env());
    }

    Napi::Value SetLanguage(const Napi::CallbackInfo& info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    Napi::Value SetDecodeOptions(const Napi::CallbackInfo& info) {
        Napi::Error::New(info.Env(), kUnsupportedPlatformMessage)
            .ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    Napi::Value Close(const Napi::CallbackInfo& info) {
        return info.Env().Undefined();
    }

    Napi::Value GetIsClosed(const Napi::CallbackInfo& info) {
        return Napi::Boolean::New(info.Env(), false);
    }
};

Napi::FunctionReference PipelineSession::constructor;

Napi::Object PipelineSession::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "PipelineSession", {
        InstanceMethod<&PipelineSession::Push>("push"),
        InstanceMethod<&PipelineSession::Finalize>("finalize"),
        InstanceMethod<&PipelineSession::SetLanguage>("setLanguage"),
        InstanceMethod<&PipelineSession::SetDecodeOptions>("setDecodeOptions"),
        InstanceMethod<&PipelineSession::Close>("close"),
        InstanceAccessor<&PipelineSession::GetIsClosed>("isClosed"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("PipelineSession", func);
    return exports;
}

Napi::Object BuildCapabilities(Napi::Env env) {
    Napi::Object capabilities = Napi::Object::New(env);
    capabilities.Set("whisper", Napi::Boolean::New(env, true));
    capabilities.Set("vad", Napi::Boolean::New(env, true));
    capabilities.Set("gpuDiscovery", Napi::Boolean::New(env, true));
    capabilities.Set("pipeline", Napi::Boolean::New(env, false));
    capabilities.Set("diarization", Napi::Boolean::New(env, false));
    return capabilities;
}

Napi::Value GetCapabilities(const Napi::CallbackInfo& info) {
    return BuildCapabilities(info.Env());
}

} // namespace

Napi::Object InitPipelineStubs(Napi::Env env, Napi::Object exports) {
    PipelineModel::Init(env, exports);
    PipelineSession::Init(env, exports);
    exports.Set("getCapabilities", Napi::Function::New(env, GetCapabilities));
    return exports;
}
