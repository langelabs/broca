import { runLoadRound } from "./load-test.js";

const inputs = [
    "Good morning. The community garden opens at eight.",
    "Please leave your umbrella beside the front door.",
    "The afternoon train arrives at platform three.",
    "Take a deep breath and slowly count to five.",
    "A small brown rabbit hopped over the garden wall.",
    "Put two tablespoons of olive oil in the pan.",
    "Your library books are due on Thursday.",
    "The museum closes early on the first Monday of each month.",
    "Welcome aboard. Please keep the aisle clear.",
    "I found my keys underneath the kitchen table.",
    "The next bus leaves in about fifteen minutes.",
    "Could you water the basil while I am away?",
    "On clear nights, we can see stars from the balcony.",
    "Simmer the soup until the carrots are tender.",
    "The bakery has fresh rye bread this morning.",
    "Turn left at the fountain and follow the path uphill.",
    "The meeting has moved to the room beside the elevator.",
    "Please bring a notebook and a pencil to class.",
    "We watched the tide come in from the end of the pier.",
    "The final score was twenty-three to nineteen.",
];

/**
 * Time concurrent audio requests using distinct inputs.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Round duration in milliseconds.
 */
export default function runAudioLoadTest(round, model) {
    return runLoadRound("audio", inputs, round, model);
}
