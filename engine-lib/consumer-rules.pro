# Consumers of this AAR must not strip the engine entry points or the
# sherpa-onnx native bridge classes. The engines are loaded reflectively
# via getInstance() and the sherpa-onnx classes register native methods.
-keep class com.CodeBySonu.VoxSherpa.KokoroEngine { *; }
-keep class com.CodeBySonu.VoxSherpa.VoiceEngine { *; }
# v2.8.0 (storyvox#119) — third in-process voice family. Loaded
# reflectively via getInstance() from storyvox's EnginePlayer; without
# this keep rule R8 strips the whole class because nothing else in the
# AAR references it directly.
-keep class com.CodeBySonu.VoxSherpa.KittenEngine { *; }
-keep class com.CodeBySonu.VoxSherpa.AudioEmotionHelper { *; }
-keep class com.CodeBySonu.VoxSherpa.Sonic { *; }
-keep class com.CodeBySonu.VoxSherpa.KokoroVoiceHelper { *; }
-keep class com.CodeBySonu.VoxSherpa.GenerationParams { *; }
-keep class com.k2fsa.sherpa.onnx.** { *; }
