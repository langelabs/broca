import { runLoadRound } from "./load-test.js";

const inputs = [
    "A paper boat drifts down a shallow stream, static camera.",
    "A potter shapes a small bowl on a slowly spinning wheel, close-up.",
    "A red balloon sways on a string beside a park bench.",
    "Alternating waves wash over smooth pebbles on a quiet beach.",
    "A houseplant's leaves move gently beside an open window.",
    "Steam rises from a mug as morning light falls across the table.",
    "A bicycle rolls past a brick wall from left to right.",
    "A marble rolls down a wooden ramp and settles into a cup.",
    "A baker dusts a loaf of bread with flour, overhead shot.",
    "A slow pan across books and plants on a low wooden shelf.",
    "A kitten bats at a felt ball on a rug.",
    "A fountain sends ripples outward across a round stone basin.",
    "Clouds drift behind a white lighthouse on a clear afternoon.",
    "A blue toy train moves through a miniature railway station.",
    "Rain leaves ripples in a puddle reflecting a street lamp.",
    "A scoop of ice cream slowly melts in a glass dessert bowl.",
    "A hand turns the pages of an illustrated notebook.",
    "An empty hammock swings gently between two trees.",
    "A ceramic cup rotates on a display turntable against a gray backdrop.",
    "A curtain billows through a narrow beam of afternoon sunlight.",
];

/**
 * Time concurrent video requests using distinct inputs.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Round duration in milliseconds.
 */
export default function runVideoLoadTest(round, model) {
    return runLoadRound("video", inputs, round, model);
}
