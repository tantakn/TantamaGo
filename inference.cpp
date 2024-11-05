#include <iostream>
// ...既存のコード...
#include <onnxruntime_cxx_api.h>

int main() {
    // ...既存のコード...

    // ONNX Runtime 環境の作成
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // セッションオプションの設定
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // CUDA Execution Provider の追加
    session_options.AppendExecutionProvider_CUDA(0);

    // モデルのパスを指定
    const char* model_path = "/path/to/your_model.onnx";

    // ONNX モデルのロード
    Ort::Session session(env, model_path, session_options);

    // 入力テンソルの作成
    // ...入力データの準備...

    // 推論の実行
    // ...推論処理...

    // 結果の表示
    // ...結果の処理...

    return 0;
}
