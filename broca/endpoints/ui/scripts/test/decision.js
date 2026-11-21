import { runLoadRound } from "./load-test.js";

const decisions = [
    [{"color":"green"}, "Is the color green?"],
    [{"inventory":7,"minimum":10}, "Is inventory below the minimum?"],
    [{"weather":"drizzle","umbrella":true}, "Did the person bring an umbrella?"],
    [{"gate":"open","access":"staff"}, "Can the general public enter this gate?"],
    [{"distance_km":3,"time_budget_minutes":15}, "Can someone walking at 5 km/h arrive within the time budget?"],
    [{"day":"Sunday","opening_days":["Saturday","Sunday"]}, "Is the shop open today?"],
    [{"cart_total":48,"free_shipping_minimum":50}, "Does the order qualify for free shipping?"],
    [{"tags":["travel","rail"]}, "Is the article tagged as travel?"],
    [{"room_capacity":12,"attendees":12}, "Can all attendees fit in the room?"],
    [{"registration":"pending","deadline_passed":false}, "Has registration been confirmed?"],
    [{"duration_minutes":90,"time_available_minutes":60}, "Does the full film fit in the available time?"],
    [{"ingredients":["rice","tofu","broccoli"]}, "Is the meal vegetarian?"],
    [{"loan_due":"2026-05-05","returned_on":"2026-05-04"}, "Was the item returned by its due date?"],
    [{"battery_percent":24,"charger_available":false}, "Is a charger available?"],
    [{"title":"A Guide to City Parks","pages":72}, "Is the guide fewer than 50 pages?"],
    [{"reserved_seats":6,"group_size":5}, "Are there enough reserved seats for the group?"],
    [{"completion":[true,true,false]}, "Are all steps complete?"],
    [{"weight_kg":1.1,"limit_kg":2}, "Is the parcel within the weight limit?"],
    [{"payment_method":"cash","accepted":["card"]}, "Is the selected payment method accepted?"],
    [{"departure":"14:20","arrival":"15:05"}, "Is the scheduled trip longer than half an hour?"],
];

const inputs = decisions.map(([state, instructions]) => JSON.stringify({
    state,
    decision: {
        kind: "noul",
        instructions,
        yes_description: "The answer to the question is yes.",
        no_description: "The answer to the question is no.",
    },
}));

/**
 * Time concurrent decision requests using distinct inputs.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Round duration in milliseconds.
 */
export default function runDecisionLoadTest(round, model) {
    return runLoadRound("decision", inputs, round, model);
}
