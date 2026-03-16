import { homedir } from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import engramPlugin, { resolveAgentsFromConfig } from "../index.js";

describe("resolveAgentsFromConfig", () => {
  it("returns an empty array when no agents are configured", () => {
    const resolved = resolveAgentsFromConfig({});
    expect(resolved).toEqual([]);
  });

  it("resolves workspace and agentDir defaults", () => {
    const resolved = resolveAgentsFromConfig({
      agents: {
        list: [{ id: "alpha" }],
      },
    });

    expect(resolved).toEqual([
      {
        id: "alpha",
        workspace: path.join(homedir(), ".openclaw", "workspace"),
        agentDir: path.join(homedir(), ".openclaw", "agents", "alpha"),
      },
    ]);
  });

  it("uses config defaults and explicit per-agent overrides", () => {
    const resolved = resolveAgentsFromConfig({
      agents: {
        defaults: {
          workspace: "/tmp/default-workspace",
        },
        list: [
          { id: "alpha" },
          { id: "beta", workspace: "/tmp/beta-workspace", agentDir: "/tmp/beta-agent" },
        ],
      },
    });

    expect(resolved).toEqual([
      {
        id: "alpha",
        workspace: "/tmp/default-workspace",
        agentDir: path.join(homedir(), ".openclaw", "agents", "alpha"),
      },
      {
        id: "beta",
        workspace: "/tmp/beta-workspace",
        agentDir: "/tmp/beta-agent",
      },
    ]);
  });

  it("filters to a single agent", () => {
    const resolved = resolveAgentsFromConfig(
      {
        agents: {
          list: [{ id: "alpha" }, { id: "beta" }],
        },
      },
      "beta",
    );

    expect(resolved).toHaveLength(1);
    expect(resolved[0]?.id).toBe("beta");
  });

  it("throws when the requested agent is missing", () => {
    expect(() =>
      resolveAgentsFromConfig(
        {
          agents: {
            list: [{ id: "alpha" }],
          },
        },
        "missing",
      ),
    ).toThrow('Agent "missing" not found in config');
  });
});

describe("engramPlugin registerCli", () => {
  it("registers CLI under memory command namespace", () => {
    const registerCli = vi.fn();
    const registerTool = vi.fn();

    (engramPlugin as any).register({
      pluginConfig: {},
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
      },
      registerTool,
      registerCli,
    });

    expect(registerCli).toHaveBeenCalledTimes(1);
    expect(typeof registerCli.mock.calls[0]?.[0]).toBe("function");
    expect(registerCli.mock.calls[0]?.[1]).toEqual({ commands: ["engram"] });
  });
});
