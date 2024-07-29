
class setup_from_toml:
    def __init__(
        self
    ):
        return
    
    def read_area_characteristics(
        self,
        area_chars_fn,
    ):    
        with codecs.open(area_chars_fn, "r", encoding="utf-8") as f:
            area_chars = toml.load(f)
        area_chars = pd.read_csv(area_chars_fn)
        return area_chars

    def setup_area_characteristics(
        self,
        area_chars_fn,
    ):
        area_chars = self.read_area_characteristics(area_chars_fn)
        self.area_chars = area_chars

 