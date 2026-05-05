"""Blind secret injection — the LLM sees aliases, not values."""

from __future__ import annotations

from typing import Any


class GantrySecrets:
    """Inject sensitive values into tool calls without exposing them to the LLM.

    The LLM is told about **aliases** (e.g. ``DB_PASS``).  When it calls a tool
    with that alias in an argument, the real value is substituted at execution
    time — the LLM never sees or echoes the actual credential.

    Aliases are also replaced inside strings, so both of these work::

        # Entire argument is the alias
        tool.ainvoke({"password": "DB_PASS"})  →  {"password": "s3cr3t"}

        # Alias embedded in a shell command
        tool.ainvoke({"command": "mysql -u root -pDB_PASS"})
        →  {"command": "mysql -u root -ps3cr3t"}

    Example::

        import os
        from gantrygraph.security import GantrySecrets

        secrets = GantrySecrets({
            "DB_PASS":  os.environ["DB_PASSWORD"],
            "API_KEY":  os.environ["OPENAI_API_KEY"],
        })

        agent = GantryEngine(
            llm=my_llm,
            tools=[...],
            secrets=secrets,
        )
        # System prompt will automatically list: "Available secret aliases: DB_PASS, API_KEY"
        # The LLM uses them by name; values are never in the context window.
    """

    def __init__(self, secrets: dict[str, str]) -> None:
        if not secrets:
            raise ValueError("GantrySecrets requires at least one secret alias.")
        self._secrets: dict[str, str] = dict(secrets)

    @property
    def aliases(self) -> list[str]:
        """Alias names to inject into the system prompt (no real values)."""
        return list(self._secrets.keys())

    def resolve(self, args: dict[str, Any]) -> dict[str, Any]:
        """Replace secret aliases in *args* with their real values.

        Performs string substitution on every ``str`` value, including
        aliases embedded inside longer strings (e.g. shell commands).
        Non-string values are passed through unchanged.
        """
        result: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str):
                resolved = v
                for alias, real_value in self._secrets.items():
                    resolved = resolved.replace(alias, real_value)
                result[k] = resolved
            else:
                result[k] = v
        return result

    def system_prompt_hint(self) -> str:
        """Return a short system message listing available aliases."""
        aliases = ", ".join(self.aliases)
        return (
            f"Secret aliases available for tool arguments: {aliases}. "
            "Pass them by name — their values are injected securely at execution time. "
            "NEVER attempt to print, log, or return secret alias values."
        )

    def __repr__(self) -> str:
        return f"GantrySecrets(aliases={self.aliases})"
