#include "StreamingSession.h"
#include "PushWorker.h"
#include "ReclusterWorker.h"
#include "FinalizeWorker.h"

Napi::FunctionReference StreamingSession::constructor;

Napi::Object StreamingSession::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "StreamingSession", {
        InstanceMethod<&StreamingSession::Push>("push"),
        InstanceMethod<&StreamingSession::Recluster>("recluster"),
        InstanceMethod<&StreamingSession::Finalize>("finalize"),
        InstanceMethod<&StreamingSession::Close>("close"),
        InstanceAccessor<&StreamingSession::GetIsClosed>("isClosed"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("StreamingSession", func);
    return exports;
}

StreamingSession::StreamingSession(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<StreamingSession>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "StreamingSession cannot be constructed directly. Use model.createStreamingSession()")
            .ThrowAsJavaScriptException();
        return;
    }

    const StreamingConfig* config =
        info[0].As<Napi::External<StreamingConfig>>().Data();

    state_ = streaming_init(*config);
    if (!state_) {
        Napi::Error::New(env, "Failed to initialize streaming state")
            .ThrowAsJavaScriptException();
        return;
    }
}

StreamingSession::~StreamingSession() {
    Cleanup();
}

void StreamingSession::Cleanup() {
    if (state_) {
        streaming_free(state_);
        state_ = nullptr;
    }
    closed_ = true;
}

Napi::Value StreamingSession::Push(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (busy_) {
        Napi::Error::New(env, "Session is busy with another operation").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "Expected Float32Array argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::TypedArray typedArr = info[0].As<Napi::TypedArray>();
    if (typedArr.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "Expected Float32Array argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Float32Array float32Arr = info[0].As<Napi::Float32Array>();
    std::vector<float> samples(float32Arr.Data(), float32Arr.Data() + float32Arr.ElementLength());

    busy_ = true;
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new PushWorker(env, this, state_, std::move(samples), deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value StreamingSession::Recluster(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (busy_) {
        Napi::Error::New(env, "Session is busy with another operation").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    busy_ = true;
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new ReclusterWorker(env, this, state_, deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value StreamingSession::Finalize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (closed_) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (busy_) {
        Napi::Error::New(env, "Session is busy with another operation").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    busy_ = true;
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new FinalizeWorker(env, this, state_, deferred);
    worker->Queue();

    return deferred.Promise();
}

Napi::Value StreamingSession::Close(const Napi::CallbackInfo& info) {
    Cleanup();
    return info.Env().Undefined();
}

Napi::Value StreamingSession::GetIsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), closed_);
}
