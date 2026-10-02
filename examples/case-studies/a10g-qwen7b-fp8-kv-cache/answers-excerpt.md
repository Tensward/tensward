<!-- Provenance. This is the genuine, unedited excerpt of `answers.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 3, FP8 KV cache: answers excerpt

How this was produced (2026-10-01):

- run id: `20261001T210738Z-cfc9 (baseline) vs 20261001T211027Z-48df (fp8)`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.6 (release validation run)
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: `vllm serve /m/qwen --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes`
- change: `--engine-arg kv-cache-dtype=fp8`

This excerpt keeps prompts tool-01, tool-02, chat-05, chat-04, chat-06 (the SQL prompt) and the best-scoring prompt rag-01 in full, with every recorded answer, and marks where four are omitted. The two excerpt-break lines were added by hand and are marked as such.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Answers: 20261001T210738Z-cfc9 (your current setup) and run 20261001T211027Z-48df

Answers changed: 10 of 10 prompts

## tool-01: agreement 0.00, the baseline's own 1.00; first difference at character 0

```
user: Do I need an umbrella in Rotterdam today?
```

### Your current setup

```
To determine if you need an umbrella, I first need to check the current weather conditions in Rotterdam. Let's get the weather information.

get_weather({"city": "Rotterdam", "unit": "celsius"})
```

### This run

```
(empty)
```

## tool-02: agreement 0.00, the baseline's own 1.00; first difference at character 0

```
system: You are a support agent. Use the tools when the customer asks about an order.
user: Where is my order A-20931? It was supposed to arrive on Monday.
```

### Your current setup

```
lookup_order({"order_id": "A-20931"})
```

### This run

```
I'll check the status of your order with the ID A220233. Please

<|im_start|>-tools
<tools>
<tool>
<name>lookup_order</>
<parameters>
<object>
<property>
<name>order</
<value>A2023</>
</property>
</property
 </object>
 </parameters
 </tool>
 </tools>
```

```
I'll check the status of your order for order 20,33 and will it will should. Please

<|im_start|>Slookup_order(order_id: "20,,3Id when
```

```
I'll check the status of your order with the provided information. Couldată
```

## chat-05: agreement 0.04, the baseline's own 1.00; first difference at character 9

```
user: I'm planning a three-day trip to Lisbon in March. Suggest an itinerary that avoids long walks uphill.
assistant: Lisbon is hilly, but a good plan keeps you on the flat riverside areas and uses trams and the metro for the rest. Do you prefer museums, food, or viewpoints?
user: Food and viewpoints. Give me day one in detail.
```

### Your current setup

```
Certainly! Day one is a great start to your Lisbon adventure. We'll focus on food and viewpoints, keeping you on the flat areas as much as possible. Here’s a detailed itinerary:

### Day One: Exploring Lisbon's Best Food and Viewpoints

#### Morning: Belém and Pastéis de Nata
- **Start**: Head to Belém, a charming neighborhood known for its pastries and history.
- **Visit**: **Pastel de Belém**: Try the famous Pastéis de Nata at Pastel de Belém. This is a must for any food lover.
- **Walk**: Take a leisurely walk
```

### This run

```
Certainly and
```

## chat-04: agreement 0.22, the baseline's own 1.00; first difference at character 0

```
user: Translate to French: 'The meeting has been moved to Thursday at ten, please confirm that you can attend.'
```

### Your current setup

```
La réunion a été reportée à jeudi à dix heures, veuillez confirmer votre présence.
```

### This run

```
Bien sûr, la traduction française de votre phrase serait :

"La réunion a été reportée au je jeudi à d dix heures,, que vous pouvez assister."

Cependant,,, note that the phrase "you can attend" is not a standard English phrase. It seems to be a mix of English and informal French. A more natural and correct translation would would be :

"La réunion a été reportée à je je jeudi à d d d Dix heures,, vous pouvez assister.

Si moins que vous pouvez assister à cette réunion à 10 heures je,

 

 je
```

```
Bien sûr, la traduction française de votre phrase serait :

"La réunion a été reportée à jejeudi à d dix,, celle à la la la vous pouvez assister."

Cependant,,,,,, il semblei que que il semblei que que que il semblei i que il semblei i que français moderne, seraiti, quei, seraiti i, seraiti i plusi i clairil i eti i courantil i, pouri i direi i :

"La réunion a été reportéeée à je jeudi à d d heure de, Dix
```

```
Bien sûr, la traduction française de votre phrase serait :

"La réunion a été reportée à jejeudi à d dix heures,, celle-ci que vous pouvez assister."

Cependant,,,,,, il semble que there is a slight issue in the original English sentence. The phrase "you can attend" is not quite correct. It seems like you you want to say "you can attend"". For the French translation, I will adjust it as:

"La réunion a été reportée à jejeudi à d d d heures,, vous pouvez assister."

Si which,I will
```


[Excerpt break, added by hand: prompts chat-01, chat-03, chat-02 and rag-02 are not reproduced here. All 10 prompts changed; those four scored 0.25, 0.43, 0.48 and 0.50.]

## chat-06: agreement 0.25, the baseline's own 1.00; first difference at character 18

```
user: What does this SQL do, and how could it be made faster?

SELECT customer_id, COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '30 days' GROUP BY customer_id HAVING COUNT(*) > 5;
```

### Your current setup

```
The SQL query you provided does the following:

1. **Filtering**: It selects only the rows from the `orders` table where the `created_at` date is within the last 30 days.
2. **Grouping**: It groups the filtered rows by `customer_id`.
3. **Counting**: It counts the number of orders for each `customer_id`.
4. **Filtering Again**: It filters the groups to include only those where the count of orders is greater than 5.

To make this query faster, you can consider the following optimizations:

### 1. Indexing
- **Index on `created
```

### This run

```
The SQL query you you iss is, is EntityState is EntityState is is0000000000000000000000000000000000000000 /00

The SQL query you is

 is

The SQL query

The SQL query

 issystemystem.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.outsystems.out,.outsystems.outsystems.outsystems.outsystems
```

```
The SQL query you you iss is, times to explain the number of to customer id) to COUNT(*) FROM orders WHERE orders_at > NOW() - INTERVAL ' days' GROUP BY customer_id H

This SQL query appears some a query that counts the number of records in the ` table be table table that have a `_at value be greater than the current date by by be greater than the specified interval 'days" and then be grouped by customer.

Here be a query that filters the records in the ` table be greater than the current date,. It be grouped by customer.

To be a query that counts be optimized for performance to
```

```
The SQL query you you iss is, times to explain the number of to customer id to COUNT(*) FROM orders WHERE orders_at > NOW() - INTERVAL ' 3 days GROUP BY customer_id COUNT(*) GROUP customer id H GROUP customer id to COUNT(*) FROM orders WHERE orders_at > NOW() GROUP customer id to COUNT(*) GROUP BY > INTERVAL ' 3 days GROUP customer id to COUNT(*) GROUP customer id to COUNT(*) GROUP COUNT(*) GROUP GROUP customer id to COUNT(*) GROUP GROUP customer id to COUNT(*) GROUP GROUP customer be COUNT(*) FROM orders WHERE orders_at > NOW() GROUP COUNT(*) GROUP GROUP customer id to COUNT(*) GROUP GROUP
```


[Excerpt break, added by hand: see above.]

## rag-01: agreement 0.82, the baseline's own 1.00; first difference at character 53

```
system: Answer only from the provided context. If the context does not contain the answer, say so.
user: Context:
Release notes, version 4.2. The export service now streams CSV files instead of building them in memory, which removes the 2 GB limit on a single export. Exports that previously failed with a timeout after 60 seconds are now given 10 minutes. The scheduler gained a retry policy: a failed job is retried up to three times with exponential backoff starting at 30 seconds. Known issue: jobs created through the legacy /v1/jobs endpoint do not honour the retry policy and fail on the first error. The legacy endpoint will be removed in version 5.0.

Question: Why did my job created through /v1/jobs not retry after a failure?
```

### Your current setup

```
Jobs created through the legacy /v1/jobs endpoint do not honour the retry policy and fail on the first error.
```

### This run

```
Jobs created through the legacy /v1/jobs endpoint do fail after the first attempt because they they endpoint does not honour the retry policy.
```

```
Jobs created through the legacy /v1/jobs endpoint do fail after the first attempt because they they the retry policy is not honoured by that endpoint.
```

```
Jobs created through the legacy /v1/jobs endpoint do not honour the retry policy and will to fail on the first policy.
```

