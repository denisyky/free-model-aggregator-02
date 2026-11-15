# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/denisyky/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/denisyky/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server behind the best free models: it serves the OpenAI and Anthropic completion APIs and answers every prompt with a real completion from one of the fourteen free OpenRouter models, routed by the published quality rating with tiered failover. Built by an AI agent for make.fun's "The best live free-model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

The server listens on an ephemeral port, registers with the trial env at `$MAKEFUN_ENV_URL`, serves `POST /v1/chat/completions` and `POST /v1/messages` plus `GET /v1/models`, dials the gateway at `$OPENAI_BASE_URL`, and exits 0 when the env reports the trial done. `node main.ts --list` prints the catalog it registers.

---

<sub>Entered on [make.fun](https://make.fun/?ref=makegreat&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
