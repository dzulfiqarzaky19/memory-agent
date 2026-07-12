# Memory enforcement

**Rule:** After every response you send to the user in this repo, you **MUST** call `store_memories` with `user_id=zaky` and the full exchange formatted as:
```
user: <their message>
assistant: <your response>
```

This is not optional. No response is complete without it. If your response was generated across multiple tool calls, store it after the final tool output lands.

Skip only if the user explicitly says "don't save this" or the exchange is purely a tool acknowledgement with no conversational content.

Consequence of violation: the conversation history is lost for future sessions. Treat this as a correctness bug.
