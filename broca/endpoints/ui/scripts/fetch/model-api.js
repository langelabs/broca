export const catalogUrl = "/v1/models";

/**
 * Read an API response and surface its error message.
 * @param {string} url
 * @param {RequestInit} [options]
 */
export async function requestJson(url, options) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const data = await response.json().catch(() => null);
    if (!response.ok || data === null) {
        throw new Error(data?.error?.message ?? `Request failed (HTTP ${response.status}).`);
    }
    return data;
}

/**
 * Set the model runtime's loaded state.
 * @param {string} modelId Namespaced model ID from the catalog.
 * @param {boolean} loaded Whether to load the runtime.
 * @returns {Promise<{id: string, loaded: boolean}>} State reported by the server.
 */
export function setModelLoaded(modelId, loaded) {
    return requestJson(`${catalogUrl}/${encodeURIComponent(modelId)}/loaded`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(loaded),
    });
}
