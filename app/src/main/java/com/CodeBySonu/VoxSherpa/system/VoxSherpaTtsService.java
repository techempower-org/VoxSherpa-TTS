package com.CodeBySonu.VoxSherpa.system;

import android.speech.tts.TextToSpeechService;
import android.speech.tts.SynthesisRequest;
import android.speech.tts.SynthesisCallback;
import android.media.AudioFormat;
import android.content.SharedPreferences;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;
import java.util.List;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Set;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

// Model and Helper Imports
import com.CodeBySonu.VoxSherpa.VoiceEngine;
import com.CodeBySonu.VoxSherpa.KokoroEngine;
import com.CodeBySonu.VoxSherpa.system.TtsLocaleHelper;
import com.CodeBySonu.VoxSherpa.KokoroVoiceHelper;
import com.CodeBySonu.VoxSherpa.AudioEmotionHelper;

public class VoxSherpaTtsService extends TextToSpeechService {

    private String _lastLoadedKokoroModel = "";
    private int _lastLoadedSpeakerId      = -1;
    private String _lastLoadedVoiceModel  = "";
    
    // Volatile flag to handle system interruption across different threads safely
    private volatile boolean isSynthesisCancelled = false;

    // Retrieves the dynamic active language from SharedPreferences
    private String getRawActiveLanguage() {
        SharedPreferences sp = getSharedPreferences("sp1", MODE_PRIVATE);
        String activeModelType = sp.getString("active_model_type", "");
        String rawLanguage = ""; 

        if ("kokoro".equals(activeModelType)) {
            int speakerId = sp.getInt("active_kokoro_speaker", 31);
            KokoroVoiceHelper.VoiceItem voice = KokoroVoiceHelper.getById(speakerId);
            if (voice != null && voice.language != null) {
                rawLanguage = voice.language; 
            }
        } else {
            String saved = sp.getString("active_language", "");
            if (saved != null && !saved.isEmpty()) {
                rawLanguage = saved;
            }
        }
        return rawLanguage;
    }

    @Override
    protected String[] onGetLanguage() {
        return TtsLocaleHelper.getTtsLanguageArray(getRawActiveLanguage());
    }

    @Override
    protected int onIsLanguageAvailable(String lang, String country, String variant) {
        String[] currentActive = TtsLocaleHelper.getTtsLanguageArray(getRawActiveLanguage());
        if (currentActive != null && currentActive[0] != null && currentActive[0].equalsIgnoreCase(lang)) {
            return TextToSpeech.LANG_COUNTRY_AVAILABLE;
        }
        return TextToSpeech.LANG_NOT_SUPPORTED;
    }

    @Override
    protected int onLoadLanguage(String lang, String country, String variant) {
        return onIsLanguageAvailable(lang, country, variant);
    }

    @Override
    public String onGetDefaultVoiceNameFor(String lang, String country, String variant) {
        String[] currentActive = TtsLocaleHelper.getTtsLanguageArray(getRawActiveLanguage());
        if (currentActive != null && currentActive[0] != null && currentActive[0].equalsIgnoreCase(lang)) {
            return "VoxSherpa_Active";
        }
        return null;
    }

    @Override
    public List<Voice> onGetVoices() {
        List<Voice> voiceList = new ArrayList<>();
        String[] currentActive = TtsLocaleHelper.getTtsLanguageArray(getRawActiveLanguage());
        
        String langCode    = (currentActive[0] != null && !currentActive[0].isEmpty()) ? currentActive[0] : "eng";
        String countryCode = (currentActive[1] != null) ? currentActive[1] : "";

        Locale locale = new Locale(langCode, countryCode);
        Set<String> emptyFeatures = new HashSet<>();
        
        Voice activeVoice = new Voice(
            "VoxSherpa_Active", 
            locale, 
            Voice.QUALITY_VERY_HIGH, 
            Voice.LATENCY_NORMAL, 
            false, 
            emptyFeatures
        );
        
        voiceList.add(activeVoice);
        return voiceList;
    }

    @Override
    public int onIsValidVoiceName(String voiceName) {
        if ("VoxSherpa_Active".equals(voiceName)) {
            return TextToSpeech.SUCCESS;
        }
        return TextToSpeech.ERROR;
    }

    @Override
    public int onLoadVoice(String voiceName) {
        return onIsValidVoiceName(voiceName);
    }

    @Override
    protected void onStop() {
        isSynthesisCancelled = true;
        
        try {
            VoiceEngine.getInstance().cancel();
            KokoroEngine.getInstance().cancel();
        } catch (Exception ignored) {
        }
    }

    @Override
    public void onDestroy() {
        try {
            VoiceEngine.getInstance().cancel();
            KokoroEngine.getInstance().cancel();
        } catch (Exception ignored) {
        }
        super.onDestroy();
    }

    @Override
    protected void onSynthesizeText(SynthesisRequest request, SynthesisCallback callback) {
        isSynthesisCancelled = false;
        boolean hasError = false; // Track errors to avoid duplicate or conflicting callback triggers
        boolean emittedAudio = false; // Track whether any PCM was actually streamed; if not, surface error()

        try {
            SharedPreferences sp = getSharedPreferences("sp1", MODE_PRIVATE);
            SharedPreferences sp3 = getSharedPreferences("sp3", MODE_PRIVATE);

            String modelType = sp.getString("active_model_type", "");
            CharSequence charText = request.getCharSequenceText();

            // 1a. Empty text — call start()+done() (no audio). The framework treats this
            // as a successful zero-length utterance, which is the correct semantic for
            // empty input. If start() itself fails, surface as error().
            if (charText == null || charText.toString().trim().isEmpty()) {
                int startResult = callback.start(22050, AudioFormat.ENCODING_PCM_16BIT, 1);
                if (startResult != TextToSpeech.SUCCESS) {
                    callback.error();
                    hasError = true;
                }
                return;
            }

            // 1b. No model loaded → surface as error() instead of start()+done() with no audio.
            // Without this, every synth call returns "successfully" but produces silence, which
            // looks like dry-running to clients (they see onStart/onDone but hear nothing). Any
            // client that watches for sub-real-time onStart cadence will detect a runaway here.
            if (modelType.isEmpty()) {
                callback.error();
                hasError = true;
                return;
            }
            
            String text = charText.toString().trim();

            boolean isPunctOn = sp3.getBoolean("smart_punct", false);
            boolean isEmotionOn = sp3.getBoolean("emotion_tags", false);

            int systemSpeechRate = request.getSpeechRate();
            int systemPitch      = request.getPitch();

            final float engineSpeed = (systemSpeechRate > 0) ? (systemSpeechRate / 100.0f) : 1.0f;
            final float enginePitch = (systemPitch > 0)      ? (systemPitch / 100.0f)      : 1.0f;

            int sampleRate = 22050;
            boolean isKokoro = modelType.equals("kokoro");

            // 2. Load Required Engine Model
            if (isKokoro) {
                KokoroEngine engine = KokoroEngine.getInstance();
                String onnx      = sp.getString("active_model", "");
                String tokens    = sp.getString("active_tokens", "");
                String voicesBin = sp.getString("active_voices_bin", "");
                int currentSpeakerId = sp.getInt("active_kokoro_speaker", 31);

                boolean needsLoad = !engine.isReady() 
                                    || !_lastLoadedKokoroModel.equals(onnx)
                                    || _lastLoadedSpeakerId != currentSpeakerId;

                if (needsLoad) {
                    if (onnx.isEmpty() || tokens.isEmpty() || voicesBin.isEmpty()) {
                        callback.error();
                        hasError = true;
                        return;
                    }

                    if (engine.isReady() && _lastLoadedKokoroModel.equals(onnx)) {
                        engine.setActiveSpeakerId(currentSpeakerId);
                    } else {
                        String loadResult = engine.loadModel(this, onnx, tokens, voicesBin);
                        if (!"Success".equals(loadResult)) {
                            callback.error();
                            hasError = true;
                            return;
                        }
                        engine.setActiveSpeakerId(currentSpeakerId);
                    }

                    _lastLoadedKokoroModel = onnx;
                    _lastLoadedSpeakerId = currentSpeakerId;
                }

                sampleRate = engine.getSampleRate();
                if (sampleRate <= 0) sampleRate = 24000;

            } else {
                VoiceEngine engine = VoiceEngine.getInstance();
                String onnx   = sp.getString("active_model", "");
                String tokens = sp.getString("active_tokens", "");

                boolean booleanNeedsLoad = !engine.isReady() || !_lastLoadedVoiceModel.equals(onnx);

                if (booleanNeedsLoad) {
                    if (onnx.isEmpty() || tokens.isEmpty()) {
                        callback.error();
                        hasError = true;
                        return;
                    }

                    String loadResult = engine.loadModel(this, onnx, tokens);
                    if (!"Success".equals(loadResult)) {
                        callback.error();
                        hasError = true;
                        return;
                    }

                    _lastLoadedVoiceModel = onnx;
                }

                sampleRate = engine.getSampleRate();
                if (sampleRate <= 0) sampleRate = 22050;
            }

            // 3. Start Audio Stream with the Android System Framework
            int startResult = callback.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1);
            if (startResult != TextToSpeech.SUCCESS) {
                callback.error();
                hasError = true;
                return;
            }

            // 4. Sentence Splitting for Memory Efficiency
            List<String> sentences = new ArrayList<>();
            String[] parts = text.split("(?<=[.!?\\n|।])\\s+");
            for (String part : parts) {
                if (!part.trim().isEmpty()) sentences.add(part.trim());
            }
            if (sentences.isEmpty()) sentences.add(text);

            // 5. Generate Audio and Stream to System OS — PIPELINED.
            //
            // Without pipelining, every sub-sentence introduces an audible gap
            // equal to the neural-inference time for the NEXT sub-sentence
            // (typical ~100–500ms with Piper). The synthesis thread spends that
            // time generating PCM while the audio framework's buffer drains
            // empty → users hear silence between clauses.
            //
            // Pipeline: kick off generation of sub N+1 on a single-threaded
            // executor while we stream sub N (which blocks on audioAvailable
            // waiting for the framework to consume the buffer). By the time
            // streaming finishes, the next PCM is usually ready, so the only
            // gap is the audioAvailable transition between buffers, not the
            // full generation pass.
            //
            // The pool is single-threaded because the underlying engines
            // (KokoroEngine, VoiceEngine) hold a single ONNX/sherpa-onnx
            // session that is not safe to call concurrently. Calling them
            // from the worker thread while the synthesis thread is in
            // _streamAudioChunks is fine because those phases never overlap
            // for the same engine instance.
            ExecutorService genPool = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "VoxSherpa-prefetch");
                t.setPriority(Thread.NORM_PRIORITY);
                return t;
            });
            try {
                final boolean punct = isPunctOn;
                final boolean emote = isEmotionOn;
                final boolean useKokoro = isKokoro;
                final float spd = engineSpeed;
                final float pit = enginePitch;

                Future<byte[]> nextPcm = sentences.isEmpty()
                    ? null
                    : genPool.submit(() -> generateOne(sentences.get(0), punct, emote, useKokoro, spd, pit));

                for (int i = 0; i < sentences.size(); i++) {
                    if (isSynthesisCancelled) break;

                    // Wait for the chunk that was queued last iteration (or the
                    // very first one). nextPcm always points at sentence i.
                    byte[] chunkPcm;
                    try {
                        chunkPcm = nextPcm == null ? null : nextPcm.get();
                    } catch (Exception e) {
                        chunkPcm = null;
                    }

                    // Immediately queue sentence i+1 so generation overlaps with
                    // streaming of sentence i.
                    if (i + 1 < sentences.size() && !isSynthesisCancelled) {
                        final String nextSentence = sentences.get(i + 1);
                        nextPcm = genPool.submit(
                            () -> generateOne(nextSentence, punct, emote, useKokoro, spd, pit)
                        );
                    } else {
                        nextPcm = null;
                    }

                    if (isSynthesisCancelled) break;

                    if (chunkPcm != null && chunkPcm.length > 0) {
                        StreamResult result = _streamAudioChunks(chunkPcm, callback);
                        if (result == StreamResult.PARTIAL_FAILURE) {
                            // audioAvailable() rejected a chunk mid-stream. The utterance
                            // is incomplete — surface as error rather than silent partial success.
                            callback.error();
                            hasError = true;
                            return;
                        }
                        if (result == StreamResult.COMPLETE) {
                            emittedAudio = true;
                        }
                    }
                }
            } finally {
                genPool.shutdownNow();
            }

            // If start() was called but the engine produced no PCM at all (model
            // loaded but generation silently failed for every sentence), surface
            // the failure to the client instead of returning success with silence.
            if (!emittedAudio && !isSynthesisCancelled) {
                callback.error();
                hasError = true;
            }

        } catch (Exception e) {
            callback.error();
            hasError = true;
        } finally {
            // Guarantees done() is only called if error() was NOT called.
            if (!hasError) {
                callback.done();
            }
        }
    }

    /** Outcome of streaming a single sentence's PCM to the framework. */
    private enum StreamResult {
        /** No audio was written (input was empty or first write was rejected). */
        NO_AUDIO,
        /** Some chunks streamed successfully, then a later audioAvailable() failed. */
        PARTIAL_FAILURE,
        /** All chunks delivered cleanly, or stopped because the caller cancelled. */
        COMPLETE,
    }

    // Dynamic Chunk Streaming logic mapped to OS Buffer Limits.
    // Returns one of {NO_AUDIO, PARTIAL_FAILURE, COMPLETE} so the caller can
    // distinguish a clean stream from a partial-failure (which must be surfaced
    // as callback.error()) and from an empty/dry stream (which feeds the
    // emittedAudio guard at the top level).
    private StreamResult _streamAudioChunks(byte[] pcm, SynthesisCallback callback) {
        boolean wroteAny = false;
        try {
            int maxBufferSize = callback.getMaxBufferSize();
            int chunkSize = (maxBufferSize > 0) ? maxBufferSize : 8192;

            for (int offset = 0; offset < pcm.length; offset += chunkSize) {
                if (isSynthesisCancelled) {
                    break;
                }

                int end = Math.min(offset + chunkSize, pcm.length);
                int writeStatus = callback.audioAvailable(pcm, offset, end - offset);

                if (writeStatus != TextToSpeech.SUCCESS) {
                    return wroteAny ? StreamResult.PARTIAL_FAILURE : StreamResult.NO_AUDIO;
                }
                wroteAny = true;
            }
        } catch (Exception ignored) {
            return wroteAny ? StreamResult.PARTIAL_FAILURE : StreamResult.NO_AUDIO;
        }
        return wroteAny ? StreamResult.COMPLETE : StreamResult.NO_AUDIO;
    }

    /**
     * Generate PCM for one sub-sentence on whatever thread the caller chose.
     * Routed through AudioEmotionHelper if punct/emotion features are on,
     * otherwise straight through the relevant engine. Returns null on
     * inference failure — the caller treats null as "skip this sub" and
     * the top-level emittedAudio guard catches "all subs failed".
     */
    private byte[] generateOne(
        String sentence,
        boolean punctOn,
        boolean emotionOn,
        boolean useKokoro,
        float speed,
        float pitch
    ) {
        try {
            if (punctOn || emotionOn) {
                return AudioEmotionHelper.processAndGenerate(
                    sentence, punctOn, emotionOn, speed, pitch, 1.0f
                );
            }
            if (useKokoro) {
                return KokoroEngine.getInstance().generateAudioPCM(sentence, speed, pitch);
            }
            return VoiceEngine.getInstance().generateAudioPCM(sentence, speed, pitch);
        } catch (Exception e) {
            return null;
        }
    }
}
