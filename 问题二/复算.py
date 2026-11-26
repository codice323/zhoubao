# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Codex；模型系列：GPT-6；开发机构：OpenAI；具体版本发布日期未提供。
"""python 复算.py [附件.xlsx]；依赖 pandas、numpy、openpyxl。结果写入脚本所在目录。"""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
book = Path(sys.argv[1]) if len(sys.argv) > 1 else next(out.parent.glob("统一赛题*/附件.xlsx"))
p = pd.read_excel(book, sheet_name="01-附件1产品库", header=1).set_index("产品编号").sort_index()
p.columns = ["型号","机型","厂商","自由度","负载","速度","续航","误差","感知","泛化","IP","防爆",
             "单价","年运维","质保","TRL","产能"]
s = pd.read_excel(book, sheet_name="02-附件2站点需求", header=1).set_index("站点编号").sort_index()
# 原始表头 J–T 与内容错位；沿用问题一按实际内容确认的顺序。
s.columns = ["场景","名称","区域","经度","纬度","面积","任务量","人力","风险","人工成本","时段","台数上限",
             "最低负载","最低自由度","最低续航","最大误差","最低泛化","最低适配","接受机型"]
matrix = pd.read_excel(book, sheet_name="03-附件3适配矩阵", header=1).set_index("站点编号")
rules = pd.read_excel(book, sheet_name="04-附件3b危险约束", header=1).set_index("危险等级")
assert list(p.index) == [f"R{i:03}" for i in range(1,49)]
assert list(s.index) == [f"S{i:03}" for i in range(1,37)]
assert list(rules.index) == [f"R{i}" for i in range(1,6)]
assert matrix.index.is_unique and set(matrix.index) == set(s.index)
assert set(matrix.columns) == set(p.index) | {"场景类别"}
matrix = matrix.loc[s.index,p.index]
assert (matrix.eq(-1) | (matrix.ge(0) & matrix.le(100))).all().all()
assert s["风险"].isin(rules.index).all() and s["最低适配"].between(0,100).all()
assert all(set(k.split(";")) <= set(p["机型"]) for k in s["接受机型"])

# 防尘、防水两位分别比较；附件仅列明 R5 全部强制防爆，R4 的条件性要求待逐站核验。
hardware = pd.DataFrame(index=rules.index, columns=p.index, dtype=bool)
for risk, rule in rules.iterrows():
    need = rule["最低防护等级"]
    ip_pass = (p["IP"].str[2].astype(int) >= int(need[2])) & (p["IP"].str[3].astype(int) >= int(need[3]))
    hardware.loc[risk] = ip_pass & (p["防爆"].eq("是") if rule["是否强制防爆认证"].startswith("是") else True)

eligible = pd.DataFrame(index=s.index, columns=p.index, dtype=bool)
for sid, site in s.iterrows():
    risk = site["风险"]
    eligible.loc[sid] = (
        p["机型"].isin(site["接受机型"].split(";")) & (p["负载"] >= site["最低负载"])
        & (p["自由度"] >= site["最低自由度"]) & (p["续航"] >= site["最低续航"])
        & (p["误差"] <= site["最大误差"]) & (p["泛化"] >= site["最低泛化"])
        & (matrix.loc[sid] >= site["最低适配"]) & hardware.loc[risk]
        & (p["TRL"] >= rules.at[risk,"最低TRL"]))


def scaling(series, cost=False):
    span = series.max()-series.min()
    if span == 0:
        raise ValueError(f"指标 {series.name} 在样本中无差异，请复核指标设置")
    return 100*((series.max()-series) if cost else (series-series.min()))/span


perf = ["自由度","负载","速度","续航","误差","泛化"]
axes = pd.DataFrame(index=p.index)
axes["性能"] = pd.concat([scaling(p[c],cost=c=="误差") for c in perf],axis=1).mean(axis=1)
axes["经济"] = .4*scaling(p["单价"],True)+.4*scaling(p["年运维"],True)+.2*scaling(p["质保"])
axes["适配"] = eligible.groupby(s["场景"]).mean().mean(axis=0)*100
axes["安全"] = hardware.mean()*100
axes["成熟度"] = scaling(p["TRL"])
assert axes.notna().all().all() and axes.ge(0).all().all() and axes.le(100+1e-10).all().all()
weights = np.array([.30,.20,.25,.15,.10])
unit = axes.to_numpy()
base_scores = unit @ weights
names = p.index.to_numpy()


def order(scores):
    # 针对未显示的精度排序，数值误差范围内同分以产品编号升序。
    return np.argsort(-np.round(scores,12),axis=-1,kind="stable")


base_order = order(base_scores)
base_rank = np.argsort(base_order)+1
base_set = set(names[base_order[:15]])


def test_factors(factors):
    w = np.atleast_2d(factors)*weights
    w = w/w.sum(axis=1, keepdims=True)
    ranking = order(w @ unit.T)
    return ranking, np.argsort(ranking,axis=1)+1


random = np.random.default_rng(20261127)
draws = random.uniform(.8,1.2,size=(10000,len(weights)))
_, ranks = test_factors(draws)
retained = (ranks[:,base_order[:15]] <= 15).sum(axis=1)
corner_factors = np.array(list(product([.8,1.2],repeat=5)))
corners, _ = test_factors(corner_factors)
single = []
for i, axis in enumerate(axes.columns):
    for factor in [.8,1.2]:
        f = np.ones(len(weights))
        f[i] = factor
        indices, positions = test_factors(f)
        shortlist = set(names[indices[0,:15]])
        single.append([axis, "−20%" if factor<1 else "+20%", names[indices[0,0]],
                       len(base_set & shortlist),"、".join(sorted(shortlist-base_set)) or "—",
                       "、".join(sorted(base_set-shortlist)) or "—",np.abs(positions[0]-base_rank).max()])
single = pd.DataFrame(single,columns=["维度","相对扰动","第一名","保留款数","进入前15","退出前15","全体最大名次变化"])
result = p[["型号","机型"]].join(axes)
result["总分"] = base_scores
result["名次"] = base_rank
result["通过站数"] = eligible.sum()
result["通过场景数"] = eligible.groupby(s["场景"]).any().sum()
result["入围频率%"] = (ranks<=15).mean(axis=0)*100
result["最好名次"] = ranks.min(axis=0)
result["最差名次"] = ranks.max(axis=0)
result["通过站点"] = ["、".join(s.index[eligible[k]]) or "—" for k in p.index]
result = result.sort_values("名次")
short = result.head(15)
report_cols = ["名次","型号","机型","总分","通过站数","入围频率%","最好名次","最差名次"]
result.to_csv(out/"48款评分与站点.csv",encoding="utf-8-sig",float_format="%.8f")


def table(frame):
    return "<div class='scroll'>"+frame.to_html(border=0,float_format=lambda v:f"{v:.2f}")+"</div>"


def ids(values):
    return "、".join(values) or "无"


all_pass = eligible.any(axis=1)
short_pass = eligible[short.index].any(axis=1)
coverage = pd.DataFrame({"站点数":s.groupby("场景",sort=False).size(),
                         "全库可选站数":all_pass.groupby(s["场景"]).sum(),
                         "前15可选站数":short_pass.groupby(s["场景"]).sum(),
                         "前15可选型号数":eligible[short.index].groupby(s["场景"]).any().sum(axis=1)})
excluded = result[(result["名次"]>15)&(result["通过站数"]>0)]
assert len(result)==48 and len(short)==15 and result.index.is_unique and len(corners)==32
assert np.all((ranks<=15).sum(axis=1)==15) and np.isclose(result["入围频率%"].sum(),1500)
corner_keep = [len(set(names[ix[:15]]) & base_set) for ix in corners]
changes = []
for factors, ix in zip(corner_factors,corners):
    selected = set(names[ix[:15]])
    if selected != base_set:
        changes.append([*["上调" if f>1 else "下调" for f in factors],
                        ids(sorted(selected-base_set)),ids(sorted(base_set-selected))])
changed_corners = pd.DataFrame(changes,columns=[*axes.columns,"进入","退出"])
winner_ids, winner_counts = np.unique(names[np.argmin(ranks,axis=1)],return_counts=True)
winners = "、".join(f"{k}（{c/len(ranks):.2%}）" for k,c in sorted(zip(winner_ids,winner_counts),key=lambda kv:-kv[1]))
first = short.index[0]
weights_table = pd.DataFrame([
    ["性能","30%","自由度、负载、速度、续航、重复定位误差、智能泛化各占 1/6。误差越小越好，其余越大越好。"],
    ["经济","20%","单价、年运维费各占 40%（越低越好），质保期占 20%（越长越好）。采用附件 1 的基准报价。"],
    ["适配","25%","按七类场景等权，取每类场景内通过本站全部可核验门槛的站点比例，再取七类平均，乘 100。"],
    ["安全","15%","按附件 3b 核验 IP 两位及明确强制的防爆要求，满足的风险等级数除以 5，再乘 100。"],
    ["成熟度","10%","以产品 TRL 在 48 款中的位置计分，TRL 6、7、8、9 分别为 0、33.33、66.67、100。"]
],columns=["维度","基准权重","计算方式"]).set_index("维度")
html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题二｜候选产品综合评价与选型短名单</title><style>
*{{box-sizing:border-box}}body{{max-width:1180px;margin:auto;padding:36px 28px 70px;color:#193840;font:16px/1.75 system-ui,sans-serif}}
h1{{font-size:32px;line-height:1.35}}h2{{font-size:22px;margin:36px 0 10px;border-bottom:2px solid #39717b}}p{{max-width:95ch}}
.scroll{{overflow:auto;margin:14px 0 24px}}table{{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
td,th{{padding:5px 10px;border-bottom:1px solid #dee6e5;white-space:nowrap;text-align:right}}thead th{{background:#e8f0ef}}tbody th{{text-align:left}}
.method td:last-child{{white-space:normal;text-align:left}} details{{margin:20px 0}}summary{{cursor:pointer;font-weight:650}}small{{color:#4d6368}}
</style><main><h1>问题二｜48 款候选产品综合评价</h1>
<p>基准首选为 <b>{first}（{short.iloc[0]['型号']}）</b>，综合得分 {short.iloc[0]['总分']:.2f}，通过 {short.iloc[0]['通过站数']} 个站点的可核验门槛。前 15 名中共有 {short['通过站数'].gt(0).sum()} 款有可匹配站点，合计可选站点 {short_pass.sum()}/36。单站有机型可选表示有进一步核算台数与成本的价值。</p>
<h2>综合排序前 15 名</h2><p>总分满分 100。入围频率和最好、最差名次均来自下文的 10,000 组联合扰动。通过站数为零的型号保留其综合排名，在本批站点只能列作储备。</p>
{table(short[report_cols])}
<p>首位两款的适配、安全和成熟度分数均相同。R013 的性能略高，R019 的经济分略高，两者总分差仅 {short.iloc[0]['总分']-short.iloc[1]['总分']:.3f} 分。应急救援则由短名单中的 R045、R044 提供选择，两者均为有防爆认证的履带机型。</p>
<h2>指标和权重</h2><div class="method">{table(weights_table)}</div>
<p>采用加权加法模型，总分＝0.30 × 性能＋0.20 × 经济＋0.25 × 适配＋0.15 × 安全＋0.10 × 成熟度。性能、经济中的连续指标用全体 48 款的极差正向化：效益指标为 100(x−min)/(max−min)，成本指标为 100(max−x)/(max−min)。各维度均为 0–100 分。性能和适配合计 55%，体现城市多场景示范对技术能力与部署范围的重视；权重为本次分析假设，未使用专家打分。使用未展示的小数排序，相同时按编号升序。</p>
<p>产品对本站的机型类别、负载、自由度、续航、定位误差、泛化、适配分阈值、IP、明确的防爆及 TRL 门槛全部满足，才计为通过。适配矩阵的 −1 直接排除。矩阵连续分仅用于达标判断，以减少与其他维度的重复计分。七类场景等权，使任务量较少的高危场景也有独立权重。安全分仅描述库内有据可查的防护与防爆能力；功能安全等级、证书有效性和 R4 站点是否涉及可燃气体仍需部署前核验。</p>
<h2>短名单能服务哪些场景</h2>{table(coverage)}
<p>全库中有 {eligible.any().sum()} 款至少通过一个站点。前 15 的零匹配型号为 {ids(short.index[short['通过站数'].eq(0)])}。全库也无合格产品的站点为 {ids(s.index[~all_pass])}。使用短名单后额外失去可选产品的站点为 {ids(s.index[all_pass & ~short_pass])}。</p>
<p>有部署可能、但位于前 15 之外的型号为 {ids(excluded.index)}。后续计算采购方案时可将这些产品一并纳入比价；其逐站匹配关系见全量评分表和 CSV。</p>
<h2>权重灵敏性</h2><p>单独将一个一级权重乘以 0.8 或 1.2，其余先保持基准值，再将全部权重归一化到和为 1。例如，性能上调时实际权重为 0.36/1.06＝33.96%。组内权重、站点门槛与原始数据固定。</p>
{table(single.set_index(['维度','相对扰动']))}
<p>联合扰动将五个权重分别乘以独立均匀分布 U(0.8,1.2) 的系数，按同一方式归一化。共 10,000 次，随机种子 20261127。成为第一名的型号及频率为 {winners}。与基准前 15 相比，保留数为 {retained.min()}–{retained.max()} 款，平均 {retained.mean():.2f} 款。另行遍历五个系数各取 0.8 或 1.2 的 32 个端点组合，保留数为 {min(corner_keep)}–{max(corner_keep)} 款。入围频率表示在这组权重假设下的结果，不代表实际采购概率。</p>
<p>在随机扰动中始终入围的型号共 {result['入围频率%'].eq(100).sum()} 款。32 个端点中有 {len(changes)} 个使名单发生变动，具体组合如下。上调和下调均为归一化前的相对变化 20%。</p>
{table(changed_corners.set_index('性能')) if changes else '<p>所有端点组合均保留原名单。</p>'}
<details open><summary>全部 48 款的综合排序与分项得分</summary>{table(result[['名次','型号','机型','性能','经济','适配','安全','成熟度','总分','通过站数','入围频率%']])}</details>
<details><summary>逐型号查看通过的站点</summary>{table(result[['名次','通过站数','通过场景数','通过站点']])}</details>
<p><small>来源为赛题附件工作表 01 至 04（产品、站点、矩阵与危险约束）。站点表 J–T 按实际数据顺序对齐。数据处理与计算由 OpenAI Codex（GPT-6 系列，2026-11-27，具体版本发布日期未提供）辅助完成。随附脚本可重算本页各表。</small></p></main></html>"""
(out/"问题二分析.html").write_text(html,encoding="utf-8")
print(short[report_cols].to_string(), changed_corners.to_string(index=False), coverage.to_string(), sep="\n")
print("首位频率",winners,"平均保留",retained.mean(),"样本范围",retained.min(),retained.max(),"端点范围",min(corner_keep),max(corner_keep))
