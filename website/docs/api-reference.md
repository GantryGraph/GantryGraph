# API Reference

## Core

### GantryEngine

::: gantrygraph.engine.engine.GantryEngine

---

### AgentSuspended

::: gantrygraph.engine.engine.AgentSuspended

---

### GantryConfig

::: gantrygraph.config.GantryConfig

---

## State & Events

### GantryState

::: gantrygraph.core.state.GantryState

### GantryEvent

::: gantrygraph.core.events.GantryEvent

### PerceptionResult

::: gantrygraph.core.events.PerceptionResult

---

## Base Classes

### BasePerception

::: gantrygraph.core.base_perception.BasePerception

### BaseAction

::: gantrygraph.core.base_action.BaseAction

### BaseMCPConnector

::: gantrygraph.core.base_mcp.BaseMCPConnector

### BaseMemory

::: gantrygraph.memory.base.BaseMemory

---

## Security

### GuardrailPolicy

::: gantrygraph.security.policies.GuardrailPolicy

### WorkspacePolicy

::: gantrygraph.security.policies.WorkspacePolicy

### BudgetPolicy

::: gantrygraph.security.policies.BudgetPolicy

### WebhookApprovalCallback

::: gantrygraph.security.policies.WebhookApprovalCallback

---

## Actions

### ShellTools

::: gantrygraph.actions.shell.ShellTools

### BrowserTools

::: gantrygraph.actions.browser.BrowserTools

---

## Perception

### WebPage

::: gantrygraph.perception.web.WebPage

---

## Swarm

### GantrySupervisor

::: gantrygraph.swarm.supervisor.GantrySupervisor

### WorkerSpec

::: gantrygraph.swarm.worker.WorkerSpec

---

## MCP

### MCPClient

::: gantrygraph.mcp.client.MCPClient

### MCPToolRegistry

::: gantrygraph.mcp.registry.MCPToolRegistry

---

## Memory

### InMemoryVector

::: gantrygraph.memory.in_memory.InMemoryVector

---

## Telemetry

### OTelExporter

::: gantrygraph.telemetry.otel.OTelExporter

---

## Decorators

### gantry_tool

::: gantrygraph.tool.gantry_tool

---

## Graph primitives

These are exported from `gantrygraph` for custom graph topologies:

::: gantrygraph.engine.nodes.observe_node
::: gantrygraph.engine.nodes.think_node
::: gantrygraph.engine.nodes.act_node
::: gantrygraph.engine.nodes.review_node
::: gantrygraph.engine.nodes.should_continue
::: gantrygraph.engine.graph.build_graph
