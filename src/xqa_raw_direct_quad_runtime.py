"""Direct native input addresses for GQA4 four-request prefix, B>=1 stress scope."""
from src.xqa_raw_quad_gqa4_small_runtime import RawGQA4QuadSmallPlan


class RawGQA4DirectQuadPlan(RawGQA4QuadSmallPlan):
    def run(self):
        self.call(16,self.q.unsqueeze(1),self.unused,self.table,self.lengths,self.groups,1)
        self.call(4,self.q.unsqueeze(1),self.output.unsqueeze(1),self.table,self.lengths,self.b,3)
        return self.output
