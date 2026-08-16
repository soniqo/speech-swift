# VoiceChat Apple Reminders demo

Build the release CLI, then start Soniqo with the bundled three-tool Reminders
configuration:

```bash
./.build/release/speech voice-chat \
  --model /Users/ivan/voicechat-mlx-int5 \
  --mcp-config Examples/VoiceChatMCP/apple-reminders.json
```

The demo exposes only `list_reminders`, `create_reminder`, and
`update_reminder`. Complete writes execute immediately by default, and Soniqo
uses the real MCP result for its spoken reply. The runtime still rejects missing
required values, unknown reminder references, and duplicate writes emitted
without fresh user speech.

Use model-mediated confirmation only when explicitly wanted:

```bash
./.build/release/speech voice-chat \
  --model /Users/ivan/voicechat-mlx-int5 \
  --mcp-config Examples/VoiceChatMCP/apple-reminders.json \
  --mcp-write-policy confirm
```

The MCP process runs locally with the current user's Reminders permission. Test
with non-critical reminders first.
