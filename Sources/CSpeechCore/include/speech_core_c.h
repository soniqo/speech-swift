#ifndef SPEECH_CORE_C_H
#define SPEECH_CORE_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

typedef struct sc_pipeline_s* sc_pipeline_t;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

typedef enum {
    SC_MODE_PIPELINE = 0,
    SC_MODE_TRANSCRIBE_ONLY = 1,
    SC_MODE_ECHO = 2
} sc_mode_t;

typedef enum {
    SC_STATE_IDLE = 0,
    SC_STATE_LISTENING = 1,
    SC_STATE_TRANSCRIBING = 2,
    SC_STATE_THINKING = 3,
    SC_STATE_SPEAKING = 4
} sc_state_t;

typedef enum {
    SC_EVENT_SESSION_CREATED = 0,
    SC_EVENT_SPEECH_STARTED,
    SC_EVENT_SPEECH_ENDED,
    SC_EVENT_TRANSCRIPTION_COMPLETED,
    SC_EVENT_RESPONSE_CREATED,
    SC_EVENT_RESPONSE_AUDIO_DELTA,
    SC_EVENT_RESPONSE_DONE,
    SC_EVENT_ERROR
} sc_event_type_t;

typedef enum {
    SC_ROLE_SYSTEM = 0,
    SC_ROLE_USER = 1,
    SC_ROLE_ASSISTANT = 2,
    SC_ROLE_TOOL = 3
} sc_role_t;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

typedef struct {
    sc_event_type_t type;
    const char* text;
    const uint8_t* audio_data;
    size_t audio_data_length;
    float start_time;
    float end_time;
    float confidence;
} sc_event_t;

typedef struct {
    float vad_onset;
    float vad_offset;
    float min_speech_duration;
    float min_silence_duration;
    bool allow_interruptions;
    float interruption_recovery_timeout;
    float max_utterance_duration;
    const char* language;
    sc_mode_t mode;
} sc_config_t;

typedef struct {
    const char* text;   // valid until next transcribe call on same context
    float confidence;
    float start_time;
    float end_time;
} sc_transcription_result_t;

typedef struct {
    sc_role_t role;
    const char* content;
} sc_message_t;

// ---------------------------------------------------------------------------
// Callback types
// ---------------------------------------------------------------------------

typedef void (*sc_event_fn)(const sc_event_t* event, void* context);
typedef void (*sc_tts_chunk_fn)(const float* samples, size_t length,
                                bool is_final, void* context);
typedef void (*sc_llm_token_fn)(const char* token, bool is_final,
                                void* context);

// ---------------------------------------------------------------------------
// Interface vtables — platform provides these function pointers
// ---------------------------------------------------------------------------

typedef struct {
    void* context;
    sc_transcription_result_t (*transcribe)(
        void* ctx, const float* audio, size_t length, int sample_rate);
    int (*input_sample_rate)(void* ctx);
} sc_stt_vtable_t;

typedef struct {
    void* context;
    void (*synthesize)(void* ctx, const char* text, const char* language,
                       sc_tts_chunk_fn on_chunk, void* chunk_ctx);
    int (*output_sample_rate)(void* ctx);
    void (*cancel)(void* ctx);
} sc_tts_vtable_t;

typedef struct {
    void* context;
    float (*process_chunk)(void* ctx, const float* samples, size_t length);
    void (*reset)(void* ctx);
    int (*input_sample_rate)(void* ctx);
    size_t (*chunk_size)(void* ctx);
} sc_vad_vtable_t;

typedef struct {
    void* context;
    void (*chat)(void* ctx, const sc_message_t* messages, size_t count,
                 sc_llm_token_fn on_token, void* token_ctx);
    void (*cancel)(void* ctx);
} sc_llm_vtable_t;

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/// Create a config with default values (Silero VAD, Echo mode).
sc_config_t sc_config_default(void);

/// Create a voice pipeline.
/// @param stt     Speech-to-text vtable
/// @param tts     Text-to-speech vtable
/// @param llm     Language model vtable (NULL for Echo/TranscribeOnly modes)
/// @param vad     Voice activity detection vtable
/// @param config  Pipeline configuration
/// @param on_event Event callback
/// @param event_context Passed to on_event as second argument
/// @return Opaque pipeline handle, or NULL on failure
sc_pipeline_t sc_pipeline_create(
    sc_stt_vtable_t stt,
    sc_tts_vtable_t tts,
    sc_llm_vtable_t* llm,
    sc_vad_vtable_t vad,
    sc_config_t config,
    sc_event_fn on_event,
    void* event_context);

/// Destroy a pipeline and free all resources.
void sc_pipeline_destroy(sc_pipeline_t pipeline);

/// Start processing audio.
void sc_pipeline_start(sc_pipeline_t pipeline);

/// Stop processing and cancel in-progress work.
void sc_pipeline_stop(sc_pipeline_t pipeline);

/// Feed microphone audio samples (Float32 PCM at VAD sample rate).
void sc_pipeline_push_audio(sc_pipeline_t pipeline,
                            const float* samples, size_t count);

/// Inject text directly (bypasses STT, sent to LLM).
void sc_pipeline_push_text(sc_pipeline_t pipeline, const char* text);

/// Get current pipeline state.
sc_state_t sc_pipeline_state(sc_pipeline_t pipeline);

/// Check if the pipeline is running.
bool sc_pipeline_is_running(sc_pipeline_t pipeline);

#ifdef __cplusplus
}
#endif

#endif // SPEECH_CORE_C_H
