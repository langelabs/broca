export const MAX_LOAD_ROUND = 20;

/**
 * Dispatch a batch together and time it until every response body has finished.
 * @param {"audio" | "decision" | "embedding" | "image" | "text" | "video"} modality
 * @param {readonly string[]} inputs Prompts ordered by round.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Total elapsed milliseconds, including response transfer.
 */
export async function runLoadRound(modality, inputs, round, model) {
    if (!Number.isInteger(round) || round < 1 || round > Math.min(MAX_LOAD_ROUND, inputs.length)) {
        throw new RangeError("Choose a valid load test round.");
    }

    const startedAt = performance.now();
    const results = await Promise.allSettled(inputs.slice(0, round).map(async input => {
        const response = await fetch(`/${modality}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, input }),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => null);
            const detail = typeof error?.detail === "string" ? error.detail : `HTTP ${response.status}`;
            throw new Error(detail);
        }
        // Drain generated media without retaining it in the browser's memory.
        await response.body?.pipeTo(new WritableStream());
    }));
    const elapsed = performance.now() - startedAt;
    const failures = results.filter(result => result.status === "rejected");
    if (failures.length) {
        throw new Error(`${failures.length} of ${round} requests failed. ${failures[0].reason.message}`);
    }
    return elapsed;
}
