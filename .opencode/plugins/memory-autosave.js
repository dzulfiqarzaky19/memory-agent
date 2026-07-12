const API_BASE = process.env.MEMORY_API_URL || "http://localhost:8000";
const USER_ID = process.env.MEMORY_USER_ID || "zaky";

const seenCount = new Map();

function extractText(parts) {
  return parts
    .filter((p) => p.type === "text" && !p.synthetic && !p.ignored)
    .map((p) => p.text.trim())
    .filter(Boolean)
    .join("\n");
}

export const MemoryAutosave = async ({ client }) => {
  return {
    event: async ({ event }) => {
      if (event.type !== "session.idle") return;
      const sessionID = event.properties.sessionID;

      const { data, error } = await client.session.messages({
        path: { id: sessionID },
      });
      if (error || !data) {
        console.error("[memory-autosave] failed to fetch session messages", error);
        return;
      }

      const already = seenCount.get(sessionID) ?? 0;
      const pending = data.slice(already);
      seenCount.set(sessionID, data.length);

      const messages = pending
        .map((m) => ({ role: m.info.role, content: extractText(m.parts) }))
        .filter((m) => m.content && (m.role === "user" || m.role === "assistant"));

      if (messages.length === 0) return;

      try {
        const res = await fetch(`${API_BASE}/add`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: USER_ID, messages }),
        });
        if (!res.ok) {
          console.error("[memory-autosave] /add failed", res.status, await res.text());
        }
      } catch (err) {
        console.error("[memory-autosave] /add request error", err);
      }
    },
  };
};
