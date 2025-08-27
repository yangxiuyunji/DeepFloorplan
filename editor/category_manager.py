import hashlib

class CategoryManager:
    def __init__(self, base_categories=None):
        self.categories = list(dict.fromkeys(base_categories or []))  # 去重保持顺序
        self.color_map = {}
        for c in self.categories:
            self._ensure_color(c)

    def add_category(self, name: str):
        if name not in self.categories:
            self.categories.append(name)
        self._ensure_color(name)
        return name

    def _ensure_color(self, name: str):
        if name not in self.color_map:
            self.color_map[name] = self._stable_random_color(name)

    def _stable_random_color(self, seed_text: str):
        h = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        def adj(v): return int(60 + (v / 255) * 155)
        return (adj(r), adj(g), adj(b))

    def get_color(self, name: str):
        self._ensure_color(name)
        return self.color_map[name]
