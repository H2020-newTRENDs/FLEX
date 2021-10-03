
# module目的：

# 首先是生成hourly target temperature和hourly car-at-home profile，这两者供优化使用
# 其次是生成profile from appliance and hot water，这两者不进入优化而是对优化结果做post-adjustment。又分为两个func。
# func1： 生成家庭的activities，
# func2： 基于activities，结合appliance和hot water的技术效率等（随diffusion会变），生成profile（hot water --> useful energy demand）。
