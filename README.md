# AWS Bedrock Python Repositories — Complete Study Guide

**Archive analyzed:** `bedrock-awslabs-python.zip`  
**Scope:** 88 Python files across `bedrock`, `bedrock-agent`, `bedrock-agent-runtime`, and `bedrock-runtime`.  
**Method:** static code inspection, AST/API-call inventory, dependency review, and syntax compilation. No AWS calls were executed. All 88 Python files compile successfully.

## 1. The four Boto3 clients in one mental model

| Client | Plane | What it manages/does | Typical APIs in this archive |
|---|---|---|---|
| `bedrock` | Control plane | Foundation-model catalog and service/infrastructure metadata | `list_foundation_models`, `get_foundation_model` |
| `bedrock-runtime` | Data/runtime plane | Direct model inference, Converse, streaming, embeddings, images, documents, async video | `converse`, `converse_stream`, `invoke_model`, `invoke_model_with_response_stream`, `start_async_invoke` |
| `bedrock-agent` | Control plane | Agents, action groups, aliases, flows, flow versions, managed prompts, knowledge bases | `create_agent`, `prepare_agent`, `create_flow`, `create_prompt`, `create_knowledge_base` |
| `bedrock-agent-runtime` | Data/runtime plane | Invoke already-configured agents and flows; this service also owns native Knowledge Base retrieval APIs, though those APIs are not demonstrated here | `invoke_agent`, `invoke_flow` |

**Memory rule:** `bedrock` and `bedrock-agent` build/configure resources; `bedrock-runtime` and `bedrock-agent-runtime` execute them.

## 2. Recommended exam-first study order

1. `bedrock/hello_bedrock.py` — verify control-plane access and discover models.
2. One `converse.py` — learn the common message/request/response shape.
3. The matching `converse_stream.py` — learn streaming events.
4. `anthropic_claude/invoke_model.py` — compare native provider payloads with Converse.
5. `amazon_titan_text_embeddings/invoke_model.py` — embeddings and RAG foundations.
6. One `document_understanding.py` — direct document input versus retrieval.
7. `tool_use_demo.py` + `weather_tool.py` — application-managed tool calling loop.
8. Managed prompts: `prompt.py` → `run_prompt.py` → scenario file.
9. Flows: `flow.py`/version/alias → `playlist_flow.py` → `run_flow.py` → multi-turn flow.
10. Agents: wrapper → Lambda action handler → full agent scenario → runtime wrapper.
11. Knowledge Bases CRUD last; the archive does not contain a native end-to-end ingestion/retrieve example.
12. Image/video/provider duplicates and test code after the core concepts.

## 3. Highest-value conceptual lessons

### Converse versus InvokeModel
`Converse`/`ConverseStream` use a common Bedrock message schema across supported models. `InvokeModel`/`InvokeModelWithResponseStream` require each provider’s native JSON request and response format. The repeated provider folders are intentionally useful: they make the portability advantage of Converse visible.

### Control plane versus runtime
Creating a prompt, flow, agent, alias, or knowledge base uses `bedrock-agent`. Invoking a managed prompt uses `bedrock-runtime`; invoking an agent or flow uses `bedrock-agent-runtime`. This is the single most important organizing concept in these folders.

### Versions and aliases
Flows and agents are not invoked simply by their editable draft definitions. The examples prepare/configure resources, create versions or aliases, and then invoke through stable runtime identifiers. Managed prompts similarly create a version whose ARN is passed to `Converse` as `modelId`.

### Tool use versus Bedrock Agents
The weather demo uses direct `Converse` tool calling: the application supplies the schema, executes the external API, and sends a `toolResult` back. The full Agent scenario delegates orchestration to Bedrock Agents and connects an action group to Lambda through an OpenAPI schema and IAM/resource-based permissions.

### Direct document input versus RAG
The `document_understanding.py` files send an entire PDF directly to a model. The LangChain Knowledge Base demo retrieves indexed chunks first. The `with_document.py` file is not vector RAG: it loads the whole S3 document and injects it into every prompt.

## 4. Practical run plan

Use a separate virtual environment per repository folder because the supplied requirements pin different Boto3/Botocore versions.

```bash
# From the extracted repository root
python -m venv .venv-runtime
source .venv-runtime/bin/activate        # Windows: .venv-runtime\Scripts\activate
python -m pip install -r bedrock-runtime/requirements.txt
aws sts get-caller-identity
cd bedrock-runtime
python models/amazon_nova/amazon_nova_text/converse.py
```

Then run, in this order:

```bash
python models/amazon_nova/amazon_nova_text/converse_stream.py
python models/anthropic_claude/invoke_model.py
python models/amazon_titan_text_embeddings/invoke_model.py
python models/amazon_nova/amazon_nova_text/document_understanding.py
python cross-model-scenarios/tool_use_demo/tool_use_demo.py
```

For managed prompts and flows, create a new environment from `bedrock-agent/requirements.txt` and run from inside `bedrock-agent` so relative imports and resource paths resolve correctly.

## 5. Important caveats found in the code

- **Knowledge Base create is not runnable as-is.** It contains a placeholder OpenSearch Serverless collection ARN, vector index name, account number, and a hard-coded `us-east-1` embedding-model ARN. The OpenSearch collection/index must already exist and permissions must match.
- **The Knowledge Base scenario contains a type bug:** it concatenates strings with `len(all_kbs)` without converting the integer to text. The same pattern appears in the list path. Change to an f-string, for example `print(f"Found {len(all_kbs)} knowledge bases.")`.
- **The full agent scenario and managed-prompt scenario hard-code older model IDs.** Treat them as lifecycle examples and replace the model ID with one available to your account/region before execution.
- **`with_document.py` contains `breakpoint()` and runs immediately at import time.** Remove the breakpoint and add an `if __name__ == "__main__":` guard before using it.
- **The two `flow-conversation.py` files are duplicates.** Study one implementation, but remember it logically belongs to agent runtime invocation.
- **Relative working directory matters.** Several scripts open `example-data/...`, `config.yaml`, `scenario_resources/...`, or import sibling modules by bare name.
- **Integration tests can call real AWS.** Runtime tests marked `@pytest.mark.integ` execute model scripts as subprocesses; playlist/prompt scenario tests create resources. Do not run the entire test suite assuming it is free or fully mocked.
- **S3 setup helper has no teardown.** `upload_document.py` creates a bucket and uploads a file but does not delete either.
- **Compile success is not run readiness.** Credentials, IAM, model access, region support, S3, Lambda, OpenSearch, and resource propagation are external prerequisites.

## 6. What is missing from these repositories

For exam completeness, note that this archive does **not** provide native examples for several important operations: Knowledge Base data sources and ingestion jobs, `Retrieve`, `RetrieveAndGenerate`, guardrails, model evaluation jobs, batch inference, `CountTokens`, provisioned throughput, model customization/import, prompt caching, or direct agent trace parsing. The absence is about this archive, not the services themselves.

## 7. File-by-file catalog (all 88 Python files)

### bedrock

| File | LOC | Type | Priority | What it does | Key API / service | Exam/use-case value |
|---|---:|---|---|---|---|---|
| `bedrock/bedrock_wrapper.py` | 131 | Wrapper / demo | P1 | Wraps the Bedrock control-plane client to list all foundation models and fetch detailed metadata for one model, then prints capabilities such as modalities, streaming support, customization, and inference types. | bedrock: ListFoundationModels, GetFoundationModel | Best introduction to control plane versus runtime and model discovery. |
| `bedrock/hello_bedrock.py` | 60 | Runnable example | P1 | Smallest “hello Bedrock” program: creates a bedrock client, calls ListFoundationModels, and prints each model summary as JSON. | bedrock: ListFoundationModels | Run first to validate credentials, region, and basic SDK access. |
| `bedrock/scenarios/bedrock_studio_bootstrapper.py` | 1361 | Infrastructure scenario | P3 | Configuration-driven bootstrapper for Bedrock Studio prerequisites. It creates or updates IAM provisioning/service roles, policies and permission boundaries, optionally creates a KMS key/alias, and creates an OpenSearch Serverless encryption security policy. | IAM, KMS, STS, OpenSearch Serverless | Useful for infrastructure/IAM understanding; not an inference example and high privilege. |
| `bedrock/test/conftest.py` | 12 | Test fixture | P4 | Shared pytest fixtures and fake data objects used by tests in this folder. | pytest / botocore Stubber | Read only when modifying tests or learning request stubbing. |
| `bedrock/test/test_bedrock_wrapper.py` | 61 | Unit test | P4 | Validates bedrock wrapper behavior using pytest, mocks, and/or botocore Stubber across 2 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock/test/test_hello_bedrock.py` | 23 | Unit test | P4 | Validates hello bedrock behavior using pytest, mocks, and/or botocore Stubber across 1 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |

### bedrock-agent

| File | LOC | Type | Priority | What it does | Key API / service | Exam/use-case value |
|---|---:|---|---|---|---|---|
| `bedrock-agent/bedrock_agent_wrapper.py` | 285 | Wrapper | P1 | Encapsulates agent control-plane operations: create/delete/get/list agents, add action groups, create/delete aliases, list action groups and attached knowledge bases, and prepare an agent. | bedrock-agent: CreateAgent, CreateAgentActionGroup, CreateAgentAlias, PrepareAgent, Get/List/Delete operations | Core agent lifecycle and the distinction between DRAFT/prepared versions and aliases. |
| `bedrock-agent/flows/flow-conversation.py` | 185 | Runtime scenario | P1 | Runs a multi-turn flow. It preserves executionId across calls, detects INPUT_REQUIRED, asks the user for missing information, sends the new input to the requesting node, and continues until SUCCESS. | bedrock-agent-runtime: InvokeFlow | Important for multi-turn flows and continuation tokens/execution IDs. |
| `bedrock-agent/flows/flow.py` | 268 | API module | P1 | Implements flow lifecycle operations: create, prepare with status polling, update, delete, get, and paginated list. | bedrock-agent: CreateFlow, PrepareFlow, UpdateFlow, DeleteFlow, GetFlow, ListFlows | Core flow control-plane lifecycle. |
| `bedrock-agent/flows/flow_alias.py` | 189 | API module | P2 | Creates, updates, deletes, and lists aliases that route runtime traffic to a flow version. | bedrock-agent: CreateFlowAlias, UpdateFlowAlias, DeleteFlowAlias, ListFlowAliases | Shows deployment indirection: version first, alias second, invocation through alias. |
| `bedrock-agent/flows/flow_version.py` | 183 | API module | P2 | Creates immutable versions from a prepared flow, retrieves/deletes versions, and lists versions with pagination. | bedrock-agent: CreateFlowVersion, GetFlowVersion, DeleteFlowVersion, ListFlowVersions | Shows flow versioning and deployment lifecycle. |
| `bedrock-agent/flows/list_flows.py` | 57 | CLI utility | P3 | Command-line inventory utility that lists flows and then prints aliases and versions for each flow. | bedrock-agent list operations | Useful account-inspection tool; less important than creation/invocation paths. |
| `bedrock-agent/flows/playlist_flow.py` | 407 | End-to-end scenario | P1 | Builds a complete playlist-generation flow: input node → prompt node → output node; wires data connections, creates IAM role, prepares and versions the flow, creates an alias, invokes it interactively, and optionally cleans up. | bedrock + bedrock-agent + bedrock-agent-runtime + IAM | Best end-to-end flow example; teaches node definitions, JSONPath expressions, versions, aliases, and invocation. |
| `bedrock-agent/flows/roles.py` | 153 | IAM helper | P2 | Creates the execution role trusted by Bedrock Flows, attaches an inline policy allowing model invocation, and removes policies/role during cleanup. | IAM CreateRole, PutRolePolicy, DeleteRolePolicy, DeleteRole | Important prerequisite: flows need an execution role that can invoke the selected model. |
| `bedrock-agent/flows/run_flow.py` | 127 | Runtime helper | P1 | Invokes a deployed flow through bedrock-agent-runtime, processes response-stream events, extracts output/completion status, and provides an interactive playlist input wrapper. | bedrock-agent-runtime: InvokeFlow | Core runtime event handling for flowOutputEvent, flowCompletionEvent, and trace events. |
| `bedrock-agent/knowledge_bases/knowledge_base.py` | 476 | CRUD scenario / CLI | P2 | Implements create/get/update/delete/list for vector knowledge bases and offers a CLI scenario that creates an IAM role and an OpenSearch Serverless-backed knowledge base, updates it, lists resources, and cleans up. | bedrock-agent: Create/Get/Update/Delete/ListKnowledgeBases; IAM | Good control-plane overview, but not a complete RAG pipeline: no data source creation, ingestion job, Retrieve, or RetrieveAndGenerate. |
| `bedrock-agent/knowledge_bases/roles.py` | 165 | IAM helper | P3 | Creates and deletes the IAM role used by a knowledge base and manages its inline policies. | IAM role/policy operations | Prerequisite mechanics for Knowledge Bases. |
| `bedrock-agent/knowledge_bases/scenario_get_started_with_knowledge_bases.py` | 32 | Thin launcher | P4 | Imports and runs run_knowledge_base_scenario from knowledge_base.py; contains no additional SDK logic. | Delegates to knowledge_base.py | Read the underlying scenario instead. |
| `bedrock-agent/prompts/list_prompts.py` | 56 | CLI utility | P3 | Lists managed prompts in the current region and prints IDs, names, ARNs, descriptions, timestamps, and versions. | bedrock-agent: ListPrompts | Useful for inventory and pagination recognition. |
| `bedrock-agent/prompts/prompt.py` | 231 | API module | P1 | Manages Bedrock Prompt Management resources: creates a text prompt variant, extracts {{variables}} into inputVariables, optionally binds a model, versions the prompt, gets/deletes it, and paginates prompt listings. | bedrock-agent: CreatePrompt, CreatePromptVersion, GetPrompt, DeletePrompt, ListPrompts | Core managed-prompt lifecycle and prompt variables. |
| `bedrock-agent/prompts/run_prompt.py` | 58 | Runtime helper | P1 | Invokes a versioned managed prompt by passing its ARN as Converse modelId and supplying promptVariables, then joins returned text blocks. | bedrock-runtime: Converse | Key exam point: managed prompt versions are invoked through Bedrock Runtime. |
| `bedrock-agent/prompts/scenario_get_started_with_prompts.py` | 169 | End-to-end scenario | P1 | Creates a playlist prompt with variables, creates an immutable prompt version, invokes the version through Converse, prints the result, and optionally deletes the prompt. | bedrock-agent + bedrock-runtime | Best managed-prompt example and a clear control-plane/runtime handoff. |
| `bedrock-agent/scenario_get_started_with_agents.py` | 487 | End-to-end scenario | P1 | Creates an agent execution role, creates/prepares an agent, packages and deploys a Lambda tool, grants both IAM and Lambda resource permissions, creates an OpenAPI action group, creates an alias, chats through InvokeAgent with a stable sessionId, and cleans up resources. | bedrock-agent + bedrock-agent-runtime + Lambda + IAM | Best full agent lifecycle example; study action groups, permissions, prepare/alias, sessions, and completion streaming. |
| `bedrock-agent/scenario_resources/lambda_function.py` | 30 | Lambda tool | P1 | Action-group Lambda handler that returns current date/time in the response envelope expected by a Bedrock Agent OpenAPI action group, while preserving session attributes. | Lambda invoked by Bedrock Agent | Shows the exact agent-action Lambda request/response contract. |
| `bedrock-agent/test/conftest.py` | 126 | Test fixture | P4 | Shared pytest fixtures and fake data objects (FakeData, FakeFlowData, FakePromptRunData, FakePromptData, FakeKnowledgeBaseData) used by tests in this folder. | pytest / botocore Stubber | Read only when modifying tests or learning request stubbing. |
| `bedrock-agent/test/test_Invoke_flow.py` | 55 | Unit test | P4 | Validates Invoke flow behavior using pytest, mocks, and/or botocore Stubber across 1 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_bedrock_agent_wrapper.py` | 319 | Unit test | P4 | Validates bedrock agent wrapper behavior using pytest, mocks, and/or botocore Stubber across 10 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_flow.py` | 270 | Unit test | P4 | Validates flow behavior using pytest, mocks, and/or botocore Stubber across 6 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_flow_alias.py` | 180 | Unit test | P4 | Validates flow alias behavior using pytest, mocks, and/or botocore Stubber across 4 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_flow_conversation.py` | 30 | Unit test | P4 | Validates flow conversation behavior using pytest, mocks, and/or botocore Stubber across 1 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_flow_version.py` | 124 | Unit test | P4 | Validates flow version behavior using pytest, mocks, and/or botocore Stubber across 3 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_knowledge_base.py` | 498 | Unit test | P4 | Validates knowledge base behavior using pytest, mocks, and/or botocore Stubber across 5 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_playlist_flow.py` | 29 | Integration test | P4 | Runs the full playlist flow scenario as a subprocess with simulated user input and cleanup confirmation. | Real Bedrock/IAM/Flow operations | High-privilege and potentially billable integration test. |
| `bedrock-agent/test/test_prompt.py` | 186 | Unit test | P4 | Validates prompt behavior using pytest, mocks, and/or botocore Stubber across 4 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_run_prompt.py` | 70 | Unit test | P4 | Validates run prompt behavior using pytest, mocks, and/or botocore Stubber across 1 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent/test/test_scenario_get_started_with_prompts.py` | 28 | Integration test | P4 | Runs the managed-prompt scenario as a subprocess and checks successful output. | Real prompt creation and model invocation | Can create/delete resources and incur inference charges. |

### bedrock-agent-runtime

| File | LOC | Type | Priority | What it does | Key API / service | Exam/use-case value |
|---|---:|---|---|---|---|---|
| `bedrock-agent-runtime/bedrock_agent_runtime_wrapper.py` | 126 | Runtime wrapper | P1 | Wraps InvokeAgent and InvokeFlow. InvokeAgent concatenates streamed completion chunks; InvokeFlow sends first/continued executions, enables trace, and prints response-stream events. | bedrock-agent-runtime: InvokeAgent, InvokeFlow | Core runtime entry points and session/execution continuity. |
| `bedrock-agent-runtime/flows/flow-conversation.py` | 185 | Runtime scenario (duplicate) | P1 | Duplicate of bedrock-agent/flows/flow-conversation.py: demonstrates multi-turn flow invocation with executionId and INPUT_REQUIRED handling. | bedrock-agent-runtime: InvokeFlow | Read once; keep this copy conceptually under the runtime client. |
| `bedrock-agent-runtime/test/conftest.py` | 12 | Test fixture | P4 | Shared pytest fixtures and fake data objects used by tests in this folder. | pytest / botocore Stubber | Read only when modifying tests or learning request stubbing. |
| `bedrock-agent-runtime/test/test_bedrock_agent_runtime_wrapper.py` | 89 | Unit test | P4 | Validates bedrock agent runtime wrapper behavior using pytest, mocks, and/or botocore Stubber across 2 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |
| `bedrock-agent-runtime/test/test_flow_conversation.py` | 30 | Unit test | P4 | Validates flow conversation behavior using pytest, mocks, and/or botocore Stubber across 1 test cases. | pytest / mocks / Stubber | Useful for SDK request-shape testing, but secondary for exam study. |

### bedrock-runtime

| File | LOC | Type | Priority | What it does | Key API / service | Exam/use-case value |
|---|---:|---|---|---|---|---|
| `bedrock-runtime/cross-model-scenarios/tool_use_demo/tool_use_demo.py` | 252 | End-to-end runtime scenario | P1 | Implements the complete Converse tool-use loop: sends system prompt + toolConfig, appends assistant toolUse blocks, executes the local weather tool, returns toolResult blocks, and recursively calls the model until end_turn. | bedrock-runtime: Converse with toolConfig | One of the highest-value files for agents/tool calling without Bedrock Agents. |
| `bedrock-runtime/cross-model-scenarios/tool_use_demo/utils/tool_use_print_utils.py` | 87 | Presentation helper | P4 | Console-formatting functions for headers, Bedrock calls, tool requests, model responses, separators, and footer. | None | No Bedrock concepts beyond making the demo readable. |
| `bedrock-runtime/cross-model-scenarios/tool_use_demo/weather_tool.py` | 61 | Tool implementation | P1 | Defines the JSON input schema for Weather_Tool and calls Open-Meteo with latitude/longitude, returning structured weather data to the model. | External HTTP API; tool schema | Shows that the application—not the model—executes tools and returns results. |
| `bedrock-runtime/models/amazon_nova/amazon_nova_canvas/invoke_model.py` | 66 | Native inference example | P2 | Uses Nova Canvas native InvokeModel payload for text-to-image generation, randomizes the seed, decodes the base64 image, and writes a numbered PNG under output/. | bedrock-runtime: InvokeModel; amazon.nova-canvas-v1:0 | Native image request/response structure and binary output handling. |
| `bedrock-runtime/models/amazon_nova/amazon_nova_reel/text_to_video.py` | 126 | Async inference scenario | P2 | Starts an asynchronous Nova Reel text-to-video job, stores output in S3, polls GetAsyncInvoke until Completed/Failed, and reports the output.mp4 location. | bedrock-runtime: StartAsyncInvoke, GetAsyncInvoke; amazon.nova-reel-v1:0 | Important asynchronous inference pattern and S3 output configuration. |
| `bedrock-runtime/models/amazon_nova/amazon_nova_text/converse.py` | 41 | Converse example | P1 | Sends a standard messages array to Nova Lite with maxTokens/temperature/topP and reads output.message.content text. | bedrock-runtime: Converse; amazon.nova-lite-v1:0 | Canonical model-neutral synchronous conversation. |
| `bedrock-runtime/models/amazon_nova/amazon_nova_text/converse_stream.py` | 44 | Streaming example | P1 | Uses ConverseStream with Nova Lite and prints contentBlockDelta text as it arrives. | bedrock-runtime: ConverseStream; amazon.nova-lite-v1:0 | Canonical model-neutral streaming event loop. |
| `bedrock-runtime/models/amazon_nova/amazon_nova_text/document_understanding.py` | 54 | Multimodal document example | P1 | Loads a PDF as bytes, places a document block and text question in a Converse message, and asks Nova Lite to compare models described in the document. | bedrock-runtime: Converse; amazon.nova-lite-v1:0 | Direct document input versus RAG/Knowledge Bases. |
| `bedrock-runtime/models/amazon_titan_image_generator/invoke_model.py` | 66 | Native inference example | P2 | Uses Titan Image Generator V2 native TEXT_IMAGE payload, decodes the returned base64 image, and writes a numbered PNG. | bedrock-runtime: InvokeModel; amazon.titan-image-generator-v2:0 | Provider-specific image payload; compare with Nova Canvas and Stability. |
| `bedrock-runtime/models/amazon_titan_text/invoke_model.py` | 48 | Native inference example | P2 | Sends Titan Text Premier’s native inputText/textGenerationConfig payload and reads results[0].outputText. | bedrock-runtime: InvokeModel; amazon.titan-text-premier-v1:0 | Illustrates why native InvokeModel bodies differ by provider/model. |
| `bedrock-runtime/models/amazon_titan_text_embeddings/invoke_model.py` | 42 | Embedding example | P1 | Invokes Titan Text Embeddings V2 with inputText, reads the embedding vector and input token count, and prints vector length and sample values. | bedrock-runtime: InvokeModel; amazon.titan-embed-text-v2:0 | Core semantic-search/RAG concept: text → numerical vector. |
| `bedrock-runtime/models/anthropic_claude/converse.py` | 41 | Converse example | P1 | Standard synchronous Converse call to Claude 3 Haiku. | bedrock-runtime: Converse; anthropic.claude-3-haiku-20240307-v1:0 | Compare identical Converse shape across providers. |
| `bedrock-runtime/models/anthropic_claude/converse_async.py` | 102 | Concurrency example | P1 | Wraps synchronous ConverseStream iterators in async generators, launches several prompts with asyncio.to_thread, compares parallel and sequential execution, and demonstrates interleaved streams. | bedrock-runtime: ConverseStream + asyncio | Very useful for responsiveness, parallel requests, and client-side concurrency. |
| `bedrock-runtime/models/anthropic_claude/converse_stream.py` | 44 | Streaming example | P1 | Standard ConverseStream call to Claude 3 Haiku and prints content deltas. | bedrock-runtime: ConverseStream; anthropic.claude-3-haiku-20240307-v1:0 | Model-neutral streaming. |
| `bedrock-runtime/models/anthropic_claude/converse_stream_pdf.py` | 78 | Streaming document utility | P2 | Prompts for a local PDF path, sends PDF bytes plus a structured summarization prompt to a Claude cross-region inference profile, and streams the answer. | bedrock-runtime: ConverseStream; us.anthropic.claude-3-5-sonnet-20241022-v2:0 | Direct PDF input plus cross-region model ID; function name is misleading. |
| `bedrock-runtime/models/anthropic_claude/document_understanding.py` | 54 | Multimodal document example | P1 | Loads the sample PDF and asks Claude 3 Haiku to compare its models using a Converse document block. | bedrock-runtime: Converse | Direct document understanding. |
| `bedrock-runtime/models/anthropic_claude/invoke_model.py` | 52 | Native inference example | P1 | Uses Claude’s native anthropic_version/messages payload with InvokeModel and reads content[0].text. | bedrock-runtime: InvokeModel | Best side-by-side comparison with Converse. |
| `bedrock-runtime/models/anthropic_claude/invoke_model_with_response_stream.py` | 47 | Native streaming example | P2 | Uses Claude’s provider-specific native payload with InvokeModelWithResponseStream and decodes each event body to print text deltas. | bedrock-runtime: InvokeModelWithResponseStream | Contrast provider-native streaming with ConverseStream. |
| `bedrock-runtime/models/anthropic_claude/scenarios/claude3_chatbot_demo/utils/colors.py` | 16 | UI helper | P4 | Prints text using ANSI color codes. | None | Skip for exam study. |
| `bedrock-runtime/models/anthropic_claude/scenarios/claude3_chatbot_demo/utils/custom_logging.py` | 40 | Logging helper | P4 | Defines a colored logging formatter and logger factory used by the chatbot demos. | None | Skip for Bedrock concepts. |
| `bedrock-runtime/models/anthropic_claude/scenarios/claude3_chatbot_demo/utils/timeit.py` | 23 | Timing helper | P4 | Decorator that measures and prints function execution time. | None | Minor observability helper. |
| `bedrock-runtime/models/anthropic_claude/scenarios/claude3_chatbot_demo/utils/upload_document.py` | 99 | S3 setup helper | P3 | Creates a randomly named S3 bucket, uploads einstein_resume.pdf, and writes bucket/file values into config.yaml. | S3 CreateBucket, UploadFile | Prerequisite helper for with_document.py; creates billable/persistent resources and has no cleanup. |
| `bedrock-runtime/models/anthropic_claude/scenarios/claude3_chatbot_demo/with_document.py` | 108 | LangChain document-chat scenario | P3 | Loads a complete document from S3 with S3FileLoader, inserts it into every prompt, maintains ConversationBufferMemory, and runs an interactive Claude chat. It includes an active breakpoint() and auto-runs on import. | Bedrock Runtime through LangChain; S3 | Educational but not production RAG: no chunking/vector retrieval, and it needs cleanup before running. |
| `bedrock-runtime/models/anthropic_claude/scenarios/claude3_chatbot_demo/with_knowledgebase.py` | 117 | LangChain Knowledge Base scenario | P2 | Builds AmazonKnowledgeBasesRetriever + RetrievalQA around BedrockChat, retrieves one result per question, and runs an interactive chat using a supplied knowledge base ID. | Bedrock Runtime through LangChain; Bedrock Knowledge Bases retrieval | Shows framework-level RAG; compare with native agent-runtime Retrieve/RetrieveAndGenerate APIs, which are not present here. |
| `bedrock-runtime/models/cohere_command/command_r_invoke_model.py` | 46 | Native inference example | P2 | Invokes Cohere Command R with its native message/max_tokens/temperature/p payload and reads text from the response. | bedrock-runtime: InvokeModel; cohere.command-r-v1:0 | Provider-specific native schema. |
| `bedrock-runtime/models/cohere_command/command_r_invoke_model_with_response_stream.py` | 48 | Native streaming example | P2 | Streams Cohere Command R’s native response events and prints generated text chunks. | bedrock-runtime: InvokeModelWithResponseStream | Provider-native streaming. |
| `bedrock-runtime/models/cohere_command/converse.py` | 41 | Converse example | P2 | Standard synchronous Converse call to Cohere Command R. | bedrock-runtime: Converse | Confirms Converse portability across providers. |
| `bedrock-runtime/models/cohere_command/converse_stream.py` | 44 | Streaming example | P2 | Standard ConverseStream call to Cohere Command R. | bedrock-runtime: ConverseStream | Confirms streaming portability. |
| `bedrock-runtime/models/cohere_command/document_understanding.py` | 54 | Multimodal document example | P2 | Sends the sample PDF and comparison prompt to Cohere Command R+ through Converse. | bedrock-runtime: Converse; cohere.command-r-plus-v1:0 | Provider comparison for document input. |
| `bedrock-runtime/models/deepseek/document_understanding.py` | 62 | Multimodal document example | P2 | Sends the sample PDF to a DeepSeek cross-region inference profile, requests a comparison, and prints both reasoningContent and final text blocks when present. | bedrock-runtime: Converse; us.deepseek.r1-v1:0 | Shows reasoning-content blocks and cross-region inference IDs. |
| `bedrock-runtime/models/meta_llama/converse.py` | 41 | Converse example | P2 | Standard synchronous Converse call to Llama 3 8B Instruct. | bedrock-runtime: Converse | Model-neutral messaging. |
| `bedrock-runtime/models/meta_llama/converse_stream.py` | 44 | Streaming example | P2 | Standard ConverseStream call to Llama 3 8B Instruct. | bedrock-runtime: ConverseStream | Model-neutral streaming. |
| `bedrock-runtime/models/meta_llama/document_understanding.py` | 54 | Multimodal document example | P2 | Uses a Llama 3.1 cross-region inference profile to analyze the sample PDF through Converse. | bedrock-runtime: Converse; us.meta.llama3-1-8b-instruct-v1:0 | Document blocks and cross-region inference. |
| `bedrock-runtime/models/meta_llama/llama3_invoke_model.py` | 54 | Native inference example | P2 | Formats the prompt using Llama 3 instruction tokens, invokes the native model API, and reads generation. | bedrock-runtime: InvokeModel; meta.llama3-70b-instruct-v1:0 | Native prompt formatting responsibility. |
| `bedrock-runtime/models/meta_llama/llama3_invoke_model_with_response_stream.py` | 56 | Native streaming example | P2 | Uses Llama 3 native instruction formatting and streams generation chunks. | bedrock-runtime: InvokeModelWithResponseStream | Native streaming and provider-specific response decoding. |
| `bedrock-runtime/models/mistral_ai/converse.py` | 41 | Converse example | P2 | Standard synchronous Converse call to Mistral Large. | bedrock-runtime: Converse | Model-neutral messaging. |
| `bedrock-runtime/models/mistral_ai/converse_stream.py` | 44 | Streaming example | P2 | Standard ConverseStream call to Mistral Large. | bedrock-runtime: ConverseStream | Model-neutral streaming. |
| `bedrock-runtime/models/mistral_ai/document_understanding.py` | 54 | Multimodal document example | P2 | Sends the sample PDF to Mistral Large through a Converse document block. | bedrock-runtime: Converse | Provider comparison for document understanding. |
| `bedrock-runtime/models/mistral_ai/invoke_model.py` | 48 | Native inference example | P2 | Wraps the prompt in Mistral instruction syntax, calls InvokeModel, and reads outputs[0].text. | bedrock-runtime: InvokeModel | Provider-specific native schema. |
| `bedrock-runtime/models/mistral_ai/invoke_model_with_response_stream.py` | 51 | Native streaming example | P2 | Streams Mistral native output chunks from InvokeModelWithResponseStream. | bedrock-runtime: InvokeModelWithResponseStream | Provider-native streaming. |
| `bedrock-runtime/models/stability_ai/invoke_model.py` | 60 | Native image example | P2 | Calls Stable Image Core with a text prompt, decodes base64 image output, and saves a numbered PNG. | bedrock-runtime: InvokeModel; stability.stable-image-core-v1:1 | Third image-generation payload to compare with Nova Canvas and Titan. |
| `bedrock-runtime/test/conftest.py` | 12 | Test fixture | P4 | Shared pytest fixtures and fake data objects used by tests in this folder. | pytest / botocore Stubber | Read only when modifying tests or learning request stubbing. |
| `bedrock-runtime/test/test_converse.py` | 26 | Integration test | P4 | Runs five provider Converse scripts as subprocesses against real AWS and asserts non-empty output and zero exit code. | Real Bedrock Runtime calls | Can incur model charges; not a unit test. |
| `bedrock-runtime/test/test_document_understanding.py` | 28 | Integration test | P4 | Runs six document-understanding scripts as subprocesses against real AWS and checks successful output. | Real Bedrock Runtime calls with PDF input | Can incur charges and requires model/document support. |
| `bedrock-runtime/test/test_invoke_model.py` | 34 | Integration test | P4 | Runs native text, embedding, and image InvokeModel scripts as subprocesses; sleeps between cases to reduce throttling. | Real Bedrock Runtime calls | Can incur charges and create local image files. |
| `bedrock-runtime/test/test_nova_reel.py` | 182 | Unit test | P4 | Mocks Nova Reel async job submission/status responses and verifies success, failure, polling, and invalid-S3-URI paths. | pytest mocks | Safe logic validation; useful for async-job testing patterns. |

### root

| File | LOC | Type | Priority | What it does | Key API / service | Exam/use-case value |
|---|---:|---|---|---|---|---|
| `main.py` | 6 | Placeholder | P4 | Minimal project entry point that only prints a greeting; it does not use AWS or Bedrock. | None | Skip for exam study. |

## 8. Priority legend

- **P1:** read and preferably run; central exam/application concept.
- **P2:** useful extension or provider variation.
- **P3:** specialized infrastructure/framework/helper code; read selectively.
- **P4:** tests, presentation helpers, launchers, or placeholders; lowest study priority.

## 9. A compact architecture map

```text
Model discovery/configuration
  boto3.client("bedrock")
          |
          +--> List/Get foundation models

Direct inference
  boto3.client("bedrock-runtime")
          +--> Converse / ConverseStream
          +--> InvokeModel / native streaming
          +--> embeddings, images, documents, async video
          +--> invoke managed prompt version ARN

Managed orchestration resources
  boto3.client("bedrock-agent")
          +--> prompts / versions
          +--> flows / versions / aliases
          +--> agents / action groups / aliases
          +--> knowledge-base control plane

Run managed orchestration
  boto3.client("bedrock-agent-runtime")
          +--> InvokeAgent (sessionId keeps conversation)
          +--> InvokeFlow (executionId continues multi-turn flow)
```

## 10. Suggested hands-on exercises

1. Run the same prompt through Nova, Claude, Cohere, Llama, and Mistral using `Converse`; compare only the model ID. Then compare their native `InvokeModel` payloads.
2. Modify a streaming script to print `stopReason`, `usage`, and `metrics` after the stream completes.
3. Change the tool-use weather demo to add a second tool and observe how `toolUseId` links request and result.
4. Modify the managed prompt template and create a second version; invoke both version ARNs to understand immutability.
5. Draw the playlist flow’s three nodes and connections before running it; map every `sourceOutput` and `targetInput`.
6. Enable and inspect agent/flow trace events, but redact sensitive prompt or tool data before logging in real systems.