import { runLoadRound } from "./load-test.js";

const inputs = [
    "How do I renew a library card?",
    "Bicycles can be rented at the railway station.",
    "Yellow leaves on basil can indicate excessive watering.",
    "Find a vegetarian lentil soup recipe.",
    "The invoice is payable within thirty days.",
    "Where can visitors refill a water bottle in the park?",
    "The meeting moved from Tuesday afternoon to Wednesday morning.",
    "Difference between weather and climate.",
    "An envelope addressed to the previous tenant arrived in the mail.",
    "Battery recycling locations near the town center.",
    "The program removes duplicate names while preserving their original order.",
    "Which hiking trails are suitable for beginners?",
    "Consider using a smaller pot for a young tomato plant.",
    "The exhibition features textiles dyed with natural pigments.",
    "Calculate the average of monthly rainfall measurements.",
    "Refunds may take up to five business days to appear.",
    "How many passengers fit in the larger shuttle?",
    "Original documents should be stored in a dry place.",
    "Trains run less frequently on public holidays.",
    "A soft brush can remove crumbs from a keyboard.",
];

/**
 * Time concurrent embedding requests using distinct inputs.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Round duration in milliseconds.
 */
export default function runEmbeddingLoadTest(round, model) {
    return runLoadRound("embedding", inputs, round, model);
}
