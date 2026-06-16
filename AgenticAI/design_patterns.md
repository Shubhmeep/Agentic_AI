# Workflow Design Patterns

## 1. Prompt Chaining

Prompt chaining is a workflow pattern where a complex task is decomposed into a fixed sequence of smaller subtasks. Each step is handled by a separate prompt or model call, and the output from one step becomes the input for the next step.

Instead of asking the model to solve everything in one large prompt, prompt chaining creates a pipeline:

```text
Input -> LLM 1 -> Gate/Check -> LLM 2 -> LLM 3 -> Output
```

### Why Use Prompt Chaining?

Prompt chaining is useful when a task has clear stages that can be solved independently. It improves reliability because each model call has a narrower responsibility.

Common benefits:

- Breaks a large task into manageable steps.
- Makes intermediate outputs easier to inspect and debug.
- Allows validation or gating between steps.
- Reduces the chance that one prompt has to handle too many instructions at once.
- Makes the workflow easier to modify by replacing or improving one step at a time.

### Example

Suppose we want to generate a polished summary from a long article. A prompt chain might look like this:

1. Extract the key facts from the article.
2. Check whether the extracted facts are relevant and complete.
3. Convert the facts into a structured outline.
4. Write the final summary from the outline.

Each step has a specific job. The final summary is better controlled because it is based on verified intermediate outputs rather than a single broad request.

### Gate Step

A gate is an optional validation step between model calls. It checks whether the previous output is good enough to continue.

For example, a gate might verify that:

- Required fields are present.
- The answer follows the expected format.
- The output does not contain unsupported claims.
- The result is relevant to the original task.

If the gate fails, the workflow can retry the previous step, ask for correction, or stop with an error.

### When To Use It

Use prompt chaining when:

- The task naturally breaks into stages.
- Accuracy depends on intermediate reasoning or transformation.
- You need structured outputs at multiple points.
- You want to inspect, test, or validate each stage.
- Different prompts are better suited for different parts of the task.

Avoid prompt chaining when the task is simple enough for one direct prompt. Extra steps add latency, cost, and complexity.

### Design Tips

- Keep each prompt focused on one responsibility.
- Define the expected input and output format for every step.
- Add gates where mistakes would be expensive downstream.
- Preserve only the information needed by the next step.
- Log intermediate outputs during development so failures are easy to diagnose.
- Prefer deterministic formats such as JSON when later steps depend on structured data.

Prompt chaining is one of the simplest and most practical workflow design patterns for building reliable AI systems. It works best when the overall task can be expressed as a clear pipeline of transformations and checks.

## 2. Routing

Routing is a workflow pattern where an input is first classified or inspected, then sent to the most appropriate specialized sub-task, prompt, tool, or model.

Instead of using one general prompt for every possible input, routing separates concerns by directing different kinds of requests to different handlers.

```text
Input -> LLM Router -> LLM 1 -> Output
                    -> LLM 2 -> Output
                    -> LLM 3 -> Output
```

### Why Use Routing?

Routing is useful when inputs can belong to different categories that require different instructions, expertise, tools, or output formats.

Common benefits:

- Sends each input to the best specialized workflow.
- Keeps prompts smaller and more focused.
- Improves accuracy by avoiding one overloaded general prompt.
- Makes systems easier to extend with new routes.
- Helps separate concerns across different task types.

### Example

Suppose we are building a customer support assistant. Incoming messages might need different handling:

1. Billing questions go to a billing support prompt.
2. Technical errors go to a troubleshooting prompt.
3. Account access issues go to an authentication support prompt.

The router first decides what kind of request the user made. Then the selected specialist prompt handles the request using instructions tailored to that category.

### Router Step

The router is the decision-making step at the beginning of the workflow. It examines the input and chooses the correct destination.

A router might classify the input by:

- Intent, such as refund request, bug report, or general question.
- Domain, such as legal, medical, technical, or financial.
- Required tool, such as search, database lookup, calculator, or code execution.
- Complexity, such as simple answer versus multi-step workflow.

The router should usually produce a structured decision, such as:

```json
{
  "route": "billing",
  "confidence": 0.91,
  "reason": "The user is asking about an unexpected invoice charge."
}
```

### When To Use It

Use routing when:

- Different inputs require meaningfully different handling.
- You have specialized prompts, tools, or models for different tasks.
- A single prompt is becoming too large or hard to maintain.
- You want to isolate domain-specific instructions.
- You need predictable behavior across several request types.

Avoid routing when all inputs can be handled well by the same prompt. A router adds another step, so it should only be used when specialization improves quality or maintainability.

### Design Tips

- Define route names clearly and keep them stable.
- Give the router precise criteria for each route.
- Include a fallback route for ambiguous or unsupported inputs.
- Return structured router output so downstream code can use it reliably.
- Track routing accuracy during testing.
- Avoid too many overlapping routes, because similar categories make misrouting more likely.

Routing is especially helpful for AI systems that serve many types of requests. It keeps each downstream task focused while giving the overall system flexibility to handle varied inputs.

## 3. Parallelization

Parallelization is a workflow pattern where a task is broken into multiple subtasks that can run at the same time. Each subtask is handled independently, and the results are combined into a final output.

Instead of processing every step one after another, parallelization uses concurrent execution:

```text
Input -> Coordinator -> LLM 1 -> Aggregator -> Output
                    -> LLM 2 ->
                    -> LLM 3 ->
```

### Why Use Parallelization?

Parallelization is useful when several independent pieces of work can be completed at the same time. It can reduce latency and improve coverage because multiple model calls or tools can work on different parts of the problem simultaneously.

Common benefits:

- Reduces total completion time for independent subtasks.
- Allows multiple perspectives on the same input.
- Improves coverage by assigning different subtasks to specialized prompts.
- Makes large tasks easier to divide and manage.
- Enables comparison or voting across several outputs.

### Example

Suppose we want to review a long technical document. A parallel workflow might split the task into separate checks:

1. One model checks for factual accuracy.
2. One model checks for clarity and structure.
3. One model checks for missing requirements or risks.

All three checks run at the same time. The aggregator then combines their findings into one final review.

### Coordinator Step

The coordinator decides how to divide the input into parallel tasks. It may send the same input to several specialists or split the input into separate chunks.

A coordinator might assign work by:

- Section of a document.
- Type of analysis.
- Required skill or domain.
- Candidate solution strategy.
- Independent test or validation method.

The coordinator should make sure each parallel task has a clear responsibility and enough context to complete its work.

### Aggregator Step

The aggregator collects the results from all parallel subtasks and turns them into one coherent output.

The aggregator might:

- Merge similar findings.
- Remove duplicates.
- Resolve conflicts between outputs.
- Rank results by importance.
- Produce the final response in the required format.

For example, if three models propose different answers, the aggregator can compare the reasoning and choose the strongest answer or synthesize a better combined answer.

### When To Use It

Use parallelization when:

- Subtasks are independent and do not need each other's outputs.
- Latency matters and concurrent work can save time.
- You want multiple specialized analyses of the same input.
- You need to compare several candidate outputs.
- A task can be split into chunks and processed separately.

Avoid parallelization when each step depends on the result of the previous step. In that case, prompt chaining is usually a better fit.

### Design Tips

- Use parallelization only for tasks that are truly independent.
- Keep each parallel prompt focused and non-overlapping when possible.
- Give every worker the context it needs, but avoid sending unnecessary information.
- Design the aggregator carefully because it determines final output quality.
- Handle partial failures, such as one worker timing out or returning invalid output.
- Track cost, since running multiple model calls at once can be more expensive.

Parallelization is a strong pattern for speeding up workflows and improving breadth of analysis. It works best when a coordinator can clearly divide the task and an aggregator can reliably combine the results.

## 4. Orchestrator-Worker

The orchestrator-worker pattern is a workflow where a central orchestrator dynamically breaks a complex task into smaller subtasks, assigns those subtasks to worker model calls or tools, and then combines the results into a final answer.

Unlike simple parallelization, the subtasks are not always known ahead of time. The orchestrator decides what work is needed based on the input and may adapt the plan as intermediate results come back.

```text
Input -> Orchestrator -> LLM 1 -> Synthesizer -> Output
                      -> LLM 2 ->
                      -> LLM 3 ->
```

### Why Use Orchestrator-Worker?

Orchestrator-worker is useful for complex tasks where the system cannot rely on a fixed workflow. The orchestrator acts like a planner that decides which workers are needed for the specific request.

Common benefits:

- Handles complex tasks with unknown or changing requirements.
- Dynamically decomposes work instead of using a fixed sequence.
- Assigns specialized workers to different subtasks.
- Allows the workflow to adapt based on intermediate results.
- Produces a final answer by synthesizing multiple pieces of work.

### Example

Suppose we ask an AI system to create a complete market research report for a new product. The required subtasks may vary depending on the product and market.

The orchestrator might decide to:

1. Research the target audience.
2. Analyze competitors.
3. Identify pricing strategies.
4. Summarize risks and opportunities.
5. Ask additional workers to fill gaps if early results are incomplete.

After the workers finish, a synthesizer combines their outputs into one structured report.

### Orchestrator Step

The orchestrator is responsible for planning and task assignment. It decides what needs to be done, which workers should do it, and whether additional work is required.

An orchestrator might:

- Break the input into subtasks.
- Choose which specialist worker should handle each subtask.
- Decide whether tasks should run sequentially or in parallel.
- Inspect intermediate outputs.
- Create new subtasks when gaps are found.

The orchestrator should produce clear task instructions so each worker understands its role and expected output.

### Worker Step

Workers execute the subtasks assigned by the orchestrator. Each worker usually has a narrow responsibility.

Workers might be:

- Different prompts using the same model.
- Different models optimized for different tasks.
- Tool calls, such as search, code execution, or database lookup.
- Domain-specific agents with specialized instructions.

Workers should return structured outputs when possible so the synthesizer can combine them reliably.

### Synthesizer Step

The synthesizer combines worker results into the final output. It is similar to an aggregator, but it often has to reconcile a more flexible and varied set of results.

The synthesizer might:

- Merge worker outputs into one coherent response.
- Resolve contradictions.
- Remove duplicate information.
- Highlight uncertainty or missing information.
- Format the final answer for the user.

### When To Use It

Use orchestrator-worker when:

- The task is complex and cannot be fully decomposed ahead of time.
- Different inputs require different subtasks.
- The workflow needs planning, adaptation, or follow-up work.
- Several specialized workers may be needed.
- Intermediate results should influence what happens next.

Avoid orchestrator-worker for simple or predictable tasks. If the steps are fixed, prompt chaining or parallelization is usually easier to build, test, and maintain.

### Design Tips

- Give the orchestrator clear rules for planning and stopping.
- Define what kinds of workers are available and when to use each one.
- Use structured task descriptions for worker assignments.
- Make worker outputs easy for the synthesizer to compare and merge.
- Add safeguards to prevent endless planning or unnecessary subtasks.
- Log the orchestrator's plan and decisions for debugging.

The orchestrator-worker pattern is powerful for open-ended tasks that require dynamic planning. It works best when the orchestrator can make reliable decisions and the workers have clearly defined responsibilities.

## 5. Evaluator-Optimizer

The evaluator-optimizer pattern is a workflow where one model generates a candidate output and another model evaluates it. If the output is not good enough, the evaluator provides feedback and the generator tries again.

This creates an iterative improvement loop:

```text
Input -> LLM Generator -> LLM Evaluator -> Output
              ^                |
              |                v
              +--- Feedback ---+
```

### Why Use Evaluator-Optimizer?

Evaluator-optimizer is useful when output quality can be improved through critique and revision. The generator focuses on producing a solution, while the evaluator focuses on checking whether the solution meets the required criteria.

Common benefits:

- Improves output quality through feedback loops.
- Separates generation from evaluation.
- Makes quality criteria explicit.
- Helps catch errors before returning the final answer.
- Supports iterative refinement for difficult tasks.

### Example

Suppose we want an AI system to write a high-quality project proposal.

The workflow might look like this:

1. The generator writes an initial proposal.
2. The evaluator checks it against the requirements.
3. If the proposal is weak, incomplete, or unclear, the evaluator returns specific feedback.
4. The generator revises the proposal using that feedback.
5. The loop continues until the evaluator accepts the output or a maximum number of attempts is reached.

The final result is accepted only after it satisfies the evaluation criteria.

### Generator Step

The generator creates the candidate output. It should receive the original task, any relevant context, and feedback from previous evaluation rounds.

The generator might produce:

- A written answer.
- Code.
- A plan.
- A structured data object.
- A summary, classification, or recommendation.

When revising, the generator should address the evaluator's feedback directly instead of rewriting randomly.

### Evaluator Step

The evaluator checks the generator's output against clear criteria. It decides whether the output should be accepted or rejected.

An evaluator might check for:

- Correctness.
- Completeness.
- Format compliance.
- Relevance to the original request.
- Missing edge cases.
- Unsupported claims or reasoning gaps.

The evaluator should provide actionable feedback when rejecting an output.

Example evaluator response:

```json
{
  "status": "rejected",
  "feedback": "The proposal does not include a timeline or measurable success criteria.",
  "required_changes": [
    "Add a project timeline",
    "Define measurable success metrics"
  ]
}
```

### When To Use It

Use evaluator-optimizer when:

- The task has clear quality criteria.
- First drafts are often incomplete or unreliable.
- The output benefits from critique and revision.
- You need higher confidence before returning a result.
- Mistakes are easier to detect than to avoid during generation.

Avoid evaluator-optimizer when the task is simple, low-risk, or does not have a clear standard for evaluation. Iteration adds latency and cost.

### Design Tips

- Define acceptance criteria before generation begins.
- Keep evaluator feedback specific and actionable.
- Limit the maximum number of revision attempts.
- Return the best attempt with warnings if the loop cannot reach acceptance.
- Use structured evaluator output so the workflow can make reliable decisions.
- Make the evaluator stricter for high-risk outputs and lighter for low-risk tasks.

The evaluator-optimizer pattern is effective for improving quality when outputs can be checked against known standards. It works best when the evaluator can clearly explain what is wrong and the generator can use that feedback to revise the result.

## Note: Agents vs. Workflows

The five patterns above are workflow patterns. They are usually designed around a known structure: fixed steps, clear routes, planned parallel work, or controlled feedback loops.

Agents are different. An agent is more open-ended and can decide its own path through repeated interaction with an environment.

In an agent-style system:

- The path is not fixed ahead of time.
- The model may take actions, observe feedback, and choose the next step.
- The loop continues until the task is complete, a stopping condition is reached, or the system is interrupted.

```text
Human -> LLM Call -> Environment
              ^          |
              |          v
           Feedback <- Action
```

Because agents are open-ended, they introduce additional risks:

- Unpredictable path.
- Unpredictable output.
- Unpredictable costs.

For this reason, agent frameworks should be monitored carefully. Guardrails help ensure agents behave safely, consistently, and within the intended boundaries of the system.
