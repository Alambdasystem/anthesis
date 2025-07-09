class LoadingAgent:
    """
    Agent that takes some content and loads it to a destination:
    - email
    - SMS
    - file system
    - SharePoint / other APIs
    """
    def __init__(self, registry, config):
        from agents.base import BaseAgent  # Lazy import
        super().__init__(registry, name="LoadingAgent",
                         description="Loads content to email/SMS/files/etc.",
                         capabilities=["load_to_destination"])
        self.config = config  # SMTP creds, API endpoints, etc.

    def execute(self, function_name: str, **kwargs):
        from agents.base import AgentResult  # Lazy import
        if function_name == "load_to_destination":
            return self.load_to_destination(**kwargs)
        raise NotImplementedError(f"{function_name} not in LoadingAgent")

    def get_available_functions(self):
        return {"load_to_destination": "Send content to a destination"}

    def load_to_destination(self, content: str, destination: dict):
        from agents.base import AgentResult  # Lazy import
        dtype = destination["type"]
        p = destination["params"]

        if dtype == "email":
            import smtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["Subject"] = p.get("subject", "No Subject")
            msg["From"] = p["from"]
            msg["To"] = p["to"]
            msg.set_content(content)

            with smtplib.SMTP(p["smtp_host"], p["smtp_port"]) as s:
                s.starttls()
                s.login(p["username"], p["password"])
                s.send_message(msg)

            return AgentResult(success=True, output="Email sent")

        elif dtype == "file":
            path = p["path"]
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return AgentResult(success=True, output=f"Wrote file to {path}")

        return AgentResult(success=False, output=f"Unknown type {dtype}")
