# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Codex，模型系列：GPT-6，开发机构：OpenAI。使用日期：2026-11-27。具体模型版本发布日期未提供。
"""用法：python 复算.py [附件.xlsx]。依赖 pandas、openpyxl、numpy、scikit-learn、matplotlib。"""
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
book = Path(sys.argv[1]) if len(sys.argv) > 1 else next(OUT.parent.glob("统一赛题*/附件.xlsx"))


def read(sheet, columns, index):
    frame = pd.read_excel(book, sheet_name=sheet, header=1)
    frame.columns = columns
    return frame.set_index(index, verify_integrity=True)


p = read("01-附件1产品库", ["编号", "型号", "机型", "厂商", "自由度", "负载", "速度", "续航",
         "定位误差", "感知配置", "泛化", "防护", "防爆", "单价", "年运维", "质保", "TRL", "标称产能"], "编号")
# J–T 按数据实际顺序读取，源文件表头错位。
s = read("02-附件2站点需求", ["站点", "场景", "名称", "区域", "经度", "纬度", "面积", "任务量",
         "人力", "风险", "人工成本", "时段", "台数上限", "最低负载", "最低自由度", "最低续航",
         "最大误差", "最低泛化", "最低适配", "接受机型"], "站点")
m = pd.read_excel(book, sheet_name="03-附件3适配矩阵", header=1).set_index("站点编号")
assert list(p.index) == [f"R{i:03}" for i in range(1, 49)]
assert list(s.index) == [f"S{i:03}" for i in range(1, 37)]
assert set(m.index) == set(s.index) and set(m.columns[1:]) == set(p.index)
assert s["风险"].isin(["R1", "R2", "R3", "R4", "R5"]).all()
assert all(set(t.split(";")) <= set(p["机型"]) for t in s["接受机型"])
assert p.notna().all().all() and s.notna().all().all()
scores = m[p.index]
assert ((scores == -1) | ((scores >= 0) & (scores <= 100))).all().all()
policy = pd.read_excel(book, sheet_name="05-附件4政策参数", header=1).set_index("参数名称")
factor = float(policy.at["单台日均产能折算系数", "取值"])

units = {"自由度":"DOF", "负载":"kg", "速度":"m/s", "续航":"h", "定位误差":"mm", "泛化":"分",
         "标称产能":"单/日", "单价":"万元", "年运维":"万元/年", "质保":"年", "TRL":"级"}
desc = p[list(units)].describe().T.rename(columns={"mean":"均值","std":"标准差","min":"最小",
            "25%":"Q1","50%":"中位数","75%":"Q3","max":"最大"}).drop(columns="count")
desc.insert(0, "单位", pd.Series(units))
perf_cols = ["自由度", "负载", "速度", "续航", "定位误差", "泛化", "标称产能"]
v = p[perf_cols].copy()
v["定位误差"] *= -1
p["性能分"] = ((v-v.min())/(v.max()-v.min())).mean(axis=1)*100
features = pd.DataFrame({"单价":p["单价"], "性能分":p["性能分"], "防爆":p["防爆"].eq("是").astype(int)})
x = StandardScaler().fit_transform(features)
groups = KMeans(n_clusters=3, n_init=100, random_state=2026).fit_predict(x)
profiles = features.groupby(groups).mean()
special = profiles["防爆"].idxmax()
flagship = profiles.drop(special)["性能分"].idxmax()
order = ["旗舰型","性价比主力","特种专用"]
p["产品群"] = [order[2] if g == special else order[0] if g == flagship else order[1] for g in groups]
sil = silhouette_score(x, groups)
profile = p.groupby("产品群").agg(款数=("型号","size"), 均价=("单价","mean"), 均性能分=("性能分","mean"),
        均负载=("负载","mean"), 均续航=("续航","mean"), 均泛化=("泛化","mean"), 均产能=("标称产能","mean"),
        防爆款数=("防爆",lambda a: a.eq("是").sum())).loc[order]
profile["标称产能/均价"] = profile["均产能"]/profile["均价"]

rows = []
base_sets, matrix_sets, strict_sets = {}, {}, {}
for sid, r in s.iterrows():
    limits = {"负载": p["负载"] >= r["最低负载"], "自由度":p["自由度"] >= r["最低自由度"],
              "续航":p["续航"] >= r["最低续航"], "定位误差":p["定位误差"] <= r["最大误差"],
              "泛化":p["泛化"] >= r["最低泛化"]}
    accepted = p["机型"].isin(r["接受机型"].split(";"))
    base = p.index[accepted & pd.DataFrame(limits).all(axis=1)]
    score = scores.loc[sid]
    strict = base[score.loc[base].ge(r["最低适配"])]
    base_sets[sid], matrix_sets[sid], strict_sets[sid] = set(base), set(score[score>=0].index), set(strict)
    cap = r["台数上限"] * factor * p.loc[strict,"标称产能"].max() if len(strict) else 0
    reasons = []
    if not len(strict):
        # 单独哪条门槛就能排除全部可接受机型；组合性瓶颈另行标明。
        for name, mask in limits.items():
            if not (accepted & mask).any():
                reasons.append(name)
    rows.append([sid,r["场景"],r["名称"],r["风险"],r["时段"],r["任务量"],r["人力"],r["人工成本"],r["台数上限"],
                 len(base),len(matrix_sets[sid]),len(strict),cap,max(0,r["任务量"]-cap),
                 "、".join(reasons) or ("组合门槛" if not len(strict) else "—"),"、".join(strict) or "—"])
sites = pd.DataFrame(rows,columns=["站点","场景","名称","风险","时段","任务量","人力","人工成本","台数上限",
              "性能初筛款数","矩阵非负款数","复核款数","有效产能上界","缺口下界","单项瓶颈","复核通过编号"]).set_index("站点")
records = []
for scene, g in sites.groupby("场景", sort=False):
    union = lambda sets: len(set().union(*(sets[i] for i in g.index)))
    demand = g["任务量"].sum()
    records.append([scene,len(g),demand,demand/sites["任务量"].sum()*100,g["人力"].sum(),g["人工成本"].sum(),
                    *[g["风险"].eq(f"R{i}").sum() for i in range(1,6)],
                    union(base_sets),union(matrix_sets),union(strict_sets),g["复核款数"].min(),
                    demand/union(base_sets),demand/union(matrix_sets),g["缺口下界"].sum()])
a = pd.DataFrame(records, columns=["场景","站数","任务量","占比%","人力","人工成本","R1","R2","R3","R4","R5",
           "初筛并集","矩阵并集","复核并集","单站最少","初筛强度","矩阵强度","缺口下界"]).set_index("场景")

for font in ["Hiragino Sans GB","PingFang SC","Microsoft YaHei","Noto Sans CJK SC","SimHei"]:
    if font in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = font
        break
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False,"font.size":11})
colors = ["#176985","#428267","#b65f34"]


def embed(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


fig, ax = plt.subplots(figsize=(8.2,4.2),layout="constrained")
for group, color, marker in zip(order, colors, ["o","s","^"]):
    g = p[p["产品群"].eq(group)]
    ax.scatter(g["单价"],g["性能分"],s=48,label=f"{group}（{len(g)} 款）",color=color,marker=marker,alpha=.83)
for pid,offset in [("R025",(6,2)),("R007",(-15,11)),("R044",(-20,-15))]:
    ax.annotate(pid, p.loc[pid,["单价","性能分"]],xytext=offset,textcoords="offset points",fontsize=9)
ax.set(xlabel="单价（万元/台）",ylabel="通用性能分（0–100）",ylim=(0,100))
ax.legend(frameon=False, loc="upper left",ncol=3,fontsize=9)
ax.grid(axis="y",alpha=.18)
price_fig = embed(fig, "48 款机器人单价与通用性能散点图")
fig, axes = plt.subplots(1,2,figsize=(10.2,4.4),layout="constrained",sharey=True)
d = a.sort_values("任务量")
for ax,col,label,color in zip(axes,["任务量","初筛强度"],["日均任务量（单/日）","每款初筛机型对应的日均任务量"],["#176985","#b65f34"]):
    bars = ax.barh(d.index,d[col],color=color,height=.58)
    ax.bar_label(bars,fmt="%.0f" if col=="任务量" else "%.1f",padding=4,fontsize=9)
    ax.set(xlabel=label,xlim=(0,d[col].max()*1.23))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0)
demand_fig = embed(fig, "七类场景任务总量与按初筛机型数计算的供给压力")


def table(frame):
    def fmt(value):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return '<div class="scroll">'+frame.to_html(border=0, float_format=fmt, na_rep="—")+"</div>"


price = p["单价"]
zero = sites[sites["复核款数"].eq(0)]
strict_cols = ["初筛并集","矩阵并集","复核并集","单站最少","初筛强度","矩阵强度","缺口下界"]
top = a["初筛强度"].sort_values(ascending=False).iloc[:3]
ranking = "、".join(f"{name}（{v:.1f}）" for name,v in top.items())
html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题一｜候选产品与场景需求</title><style>
*{{box-sizing:border-box}}body{{max-width:1150px;margin:auto;padding:36px 30px 70px;font:16px/1.7 system-ui,sans-serif;color:#21343c}}
h1{{font-size:32px;line-height:1.3}}h2{{font-size:23px;margin:38px 0 8px;border-bottom:2px solid #266a79;padding-bottom:6px}}h3{{font-size:18px}}
img{{max-width:100%;height:auto}}.scroll{{overflow-x:auto;margin:14px 0 22px}}table{{border-collapse:collapse;white-space:nowrap;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid #dce4e5;text-align:right}}thead th{{background:#eaf1f1}}tbody th{{text-align:left}}p{{max-width:91ch}}
.note{{color:#53666e;font-size:14px}}summary{{cursor:pointer;font-weight:600}}details{{margin:20px 0}}@media print{{body{{padding:0}}.scroll{{overflow:visible}}details{{break-before:page}}}}
</style><main><h1>问题一｜候选产品与场景需求</h1>
<p>48 款产品对应 36 个站点，每日共有 {sites['任务量'].sum():,} 单任务。在附件 1、2 可直接核验的条件下，每款可选型号分摊任务量最高的三个场景依次为{ranking}。另有 {len(zero)} 个站点找不到同时满足要求的产品。</p>
<h2>一、产品价格与性能</h2><p>单价从 {price.min():.1f} 到 {price.max():.1f} 万元，最高约为最低的 {price.max()/price.min():.2f} 倍。下表按产品等权统计，标准差为样本标准差，Q1 和 Q3 为第 25 和 75 百分位。</p>{table(desc)}
<h3>三类产品的价格与能力分布</h3>{price_fig}
<p>先把自由度、负载、速度、续航、定位精度、泛化评分和标称产能按全样本极差归一化，定位误差按越小越好处理。七项等权平均后乘 100，得到用于描述的通用性能分。再把单价、通用性能分、防爆认证（有为 1，无为 0）逐列标准化，做三类 K-means 聚类。固定种子 2026，初始化 100 次，取组内平方和最小解。三类取自题意，样本平均轮廓系数为 {sil:.3f}。</p>
{table(profile)}
<p>均价单位为万元/台，均负载为 kg，均续航为 h，均产能为单/日。最后一列表示每万元购置价对应的标称日产能，按组均产能除以组均价计算。按聚类中心命名，旗舰型的通用性能均分最高，性价比主力的购置价相对低，特种专用的 {int(profile.loc['特种专用','款数'])} 款均有防爆认证。等权通用性能分未计防爆能力，特种产品的价值需要结合危险场景解释。</p>
<details><summary>展开 48 款产品的分群明细</summary>{table(p[['型号','机型','产品群','单价','年运维','性能分','防护','防爆','TRL','标称产能']])}</details>
<h2>二、站点需求结构</h2><p>任务量单位为单/日，人力为可替代人数，人工成本为现有年成本（万元/年）。R1–R5 列为各级站点数，R5 风险最高。人工成本和可替代人力均为现状数据，实际节省额取决于后续部署。</p>
{table(a[['站数','任务量','占比%','人力','人工成本','R1','R2','R3','R4','R5']].pipe(lambda t: pd.concat([t,t.sum().to_frame('合计').T])))}
<p>酒店商超服务承担 43.21% 的任务量。工业巡检的任务只占 10.50%，但可替代人力为 61 人、人工成本为 692 万元/年，两项均居首位。应急救援仅 76 单/日，4 个站点却全部处于 R4 或 R5；若只按任务量确定优先级，危险作业的替代价值会被低估。</p>
<h2>三、需求压力与站点缺口</h2>{demand_fig}
<p><b>附件 1、2 初筛。</b>按站点可接受的机型类别筛选，再逐项核对最低负载、自由度、续航、泛化评分和最高定位误差。对同一场景各站的合格产品编号取并集、去重。初筛强度＝场景日均任务量÷初筛并集款数，反映任务集中在多少种可选产品上。型号可重复采购，因此这个比值不能直接换算为缺货台数。</p>
<p><b>用附件 3、4 复核。</b>“矩阵并集”只要求适配分非负，与附件概览的缺口提示口径一致；“复核并集”要求通过初筛且达到本站适配分阈值。“单站最少”为场景内各站复核款数的最小值。矩阵的 −1 只作不可行标记。</p>{table(a[strict_cols].sort_values('初筛强度',ascending=False))}
<p>仅按矩阵非负的宽口径，酒店商超、医疗辅助、养老陪护的强度居前三。计入站点性能门槛后，酒店商超的可选型号由 31 款收窄至 6 款；科普教育由 38 款收窄至 5 款，压力升至第三。应急救援通过进一步复核的只有 3 款，全场景对少数机型的依赖尤其明显。</p>
<p><b>任务缺口下界（单/日）。</b>对每个站取通过复核产品的最高标称日产能，乘以 {factor} 和本站台数上限，得到该站产能上界。无合格机型时上界为零。逐站计算 max(0, 任务量 − 产能上界)，再按场景求和，各站的富余产能不跨站抵扣。该估算允许同一型号多台采购，暂不加预算和排班限制；标称值按双班口径使用，24 小时站点不重复翻倍，其他站点也先取此乐观上界。缺口为零仅表示此项上界检查通过。</p>
<h3>存在明确任务缺口的站点</h3>{table(zero[['场景','名称','风险','任务量','矩阵非负款数','复核款数','缺口下界','单项瓶颈']])}
<p>S008 接受四足和履带特种机型，定位误差要求 ≤1.00 mm，两类机型的最佳值仍为 1.13 mm。S030 只接受双足人形，要求续航 ≥6 h、定位误差 ≤0.60 mm；该类机型最长续航 4.8 h，最小误差 1.13 mm。单纯增加同型号台数无法弥补这两处指标差距。</p>
<p>“单项瓶颈”指仅这一项就足以排除该站接受类别下的全部候选机型。全部站点的缺口下界合计 {sites['缺口下界'].sum():.0f} 单/日。产品库能核验的安全字段有限，最终部署还需核实对应等级所要求的功能安全与认证材料。</p>
<details><summary>展开 36 个站点的需求、候选数量及型号</summary>{table(sites)}</details>
<h2>数据口径与复算</h2><p>来源为赛题配套 Excel，产品与站点描述用附件 1、2。附件 3 仅用于对照筛选口径，附件 4 提供产能折算系数。站点表 J–T 的表头与实际值错位，本次依次读取为危险等级、人工成本、服务时段、台数上限、最低负载、最低自由度、最低续航、最大定位误差、最低泛化、适配分阈值、接受机型。所有数据均在内存中读取和对齐。</p>
<p class="note">这些表与图是基于竞赛模拟数据的辅助复算材料。分析由 OpenAI Codex（GPT-6 系列，2026-11-27）辅助生成；具体模型版本发布日期未提供。随附脚本包含数据列映射、聚类输入、筛选条件和作图步骤。</p></main></html>"""
(OUT/"问题一分析.html").write_text(html,encoding="utf-8")
print(profile.to_string(), a.to_string(), zero.to_string(), sep="\n")
