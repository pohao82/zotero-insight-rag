class ChatMemory:
    def __init__(self, window_size=5):
        self.history = []
        self.window_size = window_size

    def add(self, role, message):
        self.history.append({"role": role, "content": message})
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-(self.window_size * 2):]

    def get_formatted_history(self):
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.history])
