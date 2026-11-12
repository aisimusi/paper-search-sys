// scholar-daily-dsh — DeepSeek Harness host plugin.
// Exposes the scholar-daily CLI as GUI slash commands: the user types "/" in the
// composer and picks one, or types the full command, and the handler runs the
// project's PowerShell launcher and reports its output.
import { spawn } from "node:child_process";

// The project root this plugin drives. Keep in sync if the checkout moves.
const PROJECT_ROOT = "D:\\Project\\scholar-daily";
const LAUNCHER = PROJECT_ROOT + "\\scripts\\scholar-daily.ps1";
const MAX_OUTPUT_CHARS = 20000;

function runLauncher(cmd, signal) {
  return new Promise((resolve) => {
    const child = spawn(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", LAUNCHER, cmd, "-NonInteractive"],
      { cwd: PROJECT_ROOT, env: process.env, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    let out = "";
    let err = "";
    let settled = false;
    const finish = (code, msg) => {
      if (settled) return;
      settled = true;
      resolve({ code, out, err, msg });
    };
    child.stdout.on("data", (d) => {
      out += d.toString();
    });
    child.stderr.on("data", (d) => {
      err += d.toString();
    });
    child.on("error", (e) => finish(-1, String(e)));
    child.on("close", (code) => finish(code, ""));
    if (signal) {
      signal.addEventListener(
        "abort",
        () => {
          try {
            child.kill();
          } catch {}
        },
        { once: true },
      );
    }
  });
}

function clamp(text) {
  const s = (text || "").trim();
  if (s.length <= MAX_OUTPUT_CHARS) return s;
  return "…（输出过长，仅显示末尾）\n" + s.slice(-MAX_OUTPUT_CHARS);
}

const COMMANDS = [
  { name: "scholar-doctor", description: "体检 scholar-daily（配置 + 解释器 + SDK + pdftotext）", arg: "doctor" },
  { name: "scholar-collect", description: "联网检索 + 初筛（不调用模型）", arg: "collect" },
  { name: "scholar-run", description: "完整调研（分析 + 候选 idea + 报告）", arg: "run" },
  { name: "scholar-demo", description: "离线演示（虚构样本）", arg: "demo" },
  { name: "scholar-history", description: "历史运行记录", arg: "history" },
];

const apply = Object.assign(
  (ctx) => {
    for (const c of COMMANDS) {
      ctx.commands.register({
        name: c.name,
        description: c.description,
        handler: async (invocation) => {
          const r = await runLauncher(c.arg, invocation.signal);
          const body = clamp(r.out + (r.err ? "\n[stderr]\n" + r.err : ""));
          const text = body || `(无输出，exit ${r.code})`;
          if (r.code === 0) return { kind: "success", text };
          if (r.code === 2) return { kind: "success", text: "部分失败但仍生成了报告（exit 2）\n" + text };
          const reason = r.msg ? `：${r.msg}` : "";
          return { kind: "error", text: `退出码 ${r.code}${reason}\n` + text };
        },
      });
    }
  },
  { inject: ["commands"] },
);

export default apply;
