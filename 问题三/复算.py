# 本程序及代码是在 AI 工具辅助下完成的。
# 工具：OpenAI Codex，GPT-6 系列，2026-11-27；具体版本发布日期未提供。
"""用法：python 复算.py [附件.xlsx]。依赖 numpy、pandas、openpyxl、scipy。"""
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

OUT = Path(__file__).resolve().parent
book = Path(sys.argv[1]) if len(sys.argv) > 1 else next(OUT.parent.glob("统一赛题*/附件.xlsx"))


def sheet(i):
    return pd.read_excel(book, sheet_name=i, header=1, index_col=0)


p, s, matrix, rules = [sheet(i) for i in range(1,5)]
p.columns = ["型号","机型","厂商","自由度","负载","速度","续航","误差","感知","泛化","IP","防爆","单价","年运维","质保","TRL","产能"]
# 附件 2 从 J 列起表头错位，按实际字段内容对齐。
s.columns = ["场景","名称","区域","经度","纬度","面积","需求","人力","风险","人工成本","时段","台数上限","最低负载",
             "最低自由度","最低续航","最大误差","最低泛化","最低适配","接受机型"]
cfg = pd.read_excel(book,sheet_name=5,header=1).set_index("参数名称")["取值"]
assert len(p)==48 and len(s)==36 and s.index.is_unique and p.index.is_unique
year_budget = cfg["2027 年度额度"]
net_limit = year_budget*cfg["采购支出占比上限"]
budget3 = cfg["专项资金总额"]
op_limit = cfg["首年运维预算上限"]
target = cfg["场景覆盖率下限"]
life = int(cfg["设备设计寿命"])
rate = cfg["折现率"]
site_ids = list(s.index)
scenes = list(s["场景"].unique())
scene_idx = s["场景"].map({c:i for i,c in enumerate(scenes)}).to_numpy()
risk_mask = s["风险"].isin(["R4","R5"]).to_numpy()
demand = s["需求"].to_numpy()
by_scene = np.eye(len(scenes))[scene_idx].T
scene_demand = by_scene @ demand
gross = p["单价"]*cfg["技术降价系数（2027）"]
net = gross - np.minimum(gross*cfg["2027 年购置补贴比例"],cfg["单台补贴上限"])
op = np.array([p["年运维"]*cfg["运维成本年增系数"]**a *
            (1-cfg["质保期内运维减免"]*(p["质保"]>a)) for a in range(life)])
discount = (1+rate)**(-np.arange(1,life+1,dtype=float))
base_lcc = net + discount @ op - gross*cfg["残值率"]*discount[-1]


def shifts(window):
    if window=="24小时":
        return 2
    start,end = window.split("-")
    def minutes(t):
        h,m = map(int,t.split(":"))
        return 60*h+m
    return 1 if (minutes(end)-minutes(start))%1440 <= 720 else 2


def valid_products(station, row):
    rule = rules.loc[row["风险"]]
    ip = rule["最低防护等级"]
    # 化工装置区保守按涉可燃气体处理；其他 R4 条件性认证待现场核验。
    explosion = rule["是否强制防爆认证"].startswith("是") or (row["风险"]=="R4" and "化工" in row["名称"])
    ok = (p["机型"].isin(row["接受机型"].split(";"))
        & (p["自由度"]>=row["最低自由度"]) & (p["负载"]>=row["最低负载"])
        & (p["续航"]>=row["最低续航"]) & (p["误差"]<=row["最大误差"]) & (p["泛化"]>=row["最低泛化"])
        & (p["IP"].str[2]>=ip[2]) & (p["IP"].str[3]>=ip[3])
        & (p["TRL"]>=rule["最低TRL"]) & (matrix.loc[station,p.index]>=row["最低适配"]))
    if explosion:
        ok &= p["防爆"].eq("是")
    return p.index[ok].tolist()


# 每个站点的型号组合为一个整数决策的等价编码，允许不同型号混配。
records = []
for i,(station,row) in enumerate(s.iterrows()):
    ids = valid_products(station,row)
    shift = shifts(row["时段"])
    power = cfg["单台日均能耗"]*shift/2 * 365*cfg["电价"]/10000
    coeff = cfg["单台日均产能折算系数"]*shift/2
    for n in range(int(row["台数上限"])+1):
        for combo in combinations_with_replacement(ids,n):
            pos = p.index.get_indexer(combo)
            chargers = int(np.ceil(n/cfg["充电桩配比"]))
            site_cost = cfg["单站配套改造费"]*(n>0)+cfg["充电桩单价"]*chargers
            q = min(row["需求"],p["产能"].to_numpy()[pos].sum()*coeff)
            procurement = net.to_numpy()[pos].sum()
            annual_op = op[:3,pos].sum(axis=1)
            cash = procurement+site_cost+annual_op[0]+n*power
            lcc = base_lcc.to_numpy()[pos].sum()+site_cost+n*power*discount.sum()
            records.append((i,combo,n,shift,chargers,q,procurement,annual_op[0],cash,
                            annual_op[1]+n*power,annual_op[2]+n*power,lcc,site_cost,n*power,gross.to_numpy()[pos].sum()))
options = pd.DataFrame(records,columns=["站序","组合","台数","班数","桩数","覆盖","设备净支出","首年运维","首年支出",
                                         "次年支出","第三年支出","寿命成本","配套","年电费","设备原价"])
options["三年支出"] = options[["首年支出","次年支出","第三年支出"]].sum(axis=1)
# 同站被另一组合在覆盖及全部预算、目标上支配时，可安全剪枝；完全相同保留首个。
keep = []
costcols = ["寿命成本","设备净支出","首年运维","首年支出","次年支出","第三年支出","三年支出"]
for _,g in options.groupby("站序",sort=True):
    vals = np.round(np.column_stack([-g["覆盖"],g[costcols]]),9)
    for k,v in enumerate(vals):
        better = (vals<=v).all(axis=1)
        equal = (vals==v).all(axis=1)
        if not (better & ~equal).any() and not equal[:k].any():
            keep.append(g.index[k])
options = options.loc[keep].reset_index(drop=True)
groups = [g.index.to_numpy() for _,g in options.groupby("站序",sort=True)]
stations = np.arange(len(s))
capacity = options.groupby("站序")["覆盖"].max().to_numpy()
required = np.minimum(target*scene_demand,by_scene@capacity)
risk_ceiling = capacity[risk_mask].sum()
goals = {"成本优先":.90*demand.sum(),"均衡覆盖":.95*demand.sum(),"最高覆盖":capacity.sum()}
# 每个备选组合对应的约束矩阵行，既用于群体搜索，也用于精确校验。
sid = options["站序"].to_numpy()
cov = options["覆盖"].to_numpy()
limits = np.array([net_limit,op_limit,year_budget,cfg["2028 年度额度"],cfg["2029 年度额度"],budget3])
values = np.column_stack([cov[:,None]*by_scene[:,sid].T,cov*risk_mask[sid],cov,
          options[["设备净支出","首年运维","首年支出","次年支出","第三年支出","三年支出"]]])
lcc = options["寿命成本"].to_numpy()
scale = np.r_[scene_demand,max(1,risk_ceiling),demand.sum(),limits]
site_choice = np.eye(len(s))[sid].T
A = csr_matrix(np.vstack([site_choice,values.T]))


def limits_for(goal):
    lower = np.r_[required,risk_ceiling,goal,np.full(len(limits),-np.inf)]
    upper = np.r_[np.full(len(required)+2,np.inf),limits]
    return lower,upper


def genetic(goal,seed):
    # 各基因是本站选项的行号。以最大可覆盖的低成本配置作一个种子，保留精英，锦标赛选亲。
    rng = np.random.default_rng(seed)
    population, generations = 120, 240
    starts = np.array([g[0] for g in groups])
    counts = np.array([len(g) for g in groups])
    def random_options():
        return starts+rng.integers(counts,size=(population,len(s)))
    full = np.array([min(g[np.isclose(cov[g],capacity[i])],key=lambda j:lcc[j]) for i,g in enumerate(groups)])
    pop = np.where(rng.random((population,len(s)))<.15,random_options(),full)
    pop[0] = full
    lo,hi = limits_for(goal)
    def fitness(pool):
        v = values[pool].sum(axis=1)
        violation = (np.maximum(lo-v-1e-7,0)+np.maximum(v-hi-1e-7,0))/scale
        loss = violation.sum(axis=1)
        return lcc[pool].sum(axis=1)+(loss>0)*(1e6+1e6*loss)
    for _ in range(generations):
        f = fitness(pop)
        elite = pop[np.argsort(f)[:8]].copy()
        draws = rng.integers(population,size=(2,population,4))
        winners = np.take_along_axis(draws,np.argmin(f[draws],axis=2)[:,:,None],axis=2)[:,:,0]
        pop = np.where(rng.random(pop.shape)<.5,pop[winners[0]],pop[winners[1]])
        pop = np.where(rng.random(pop.shape)<1.4/len(s),random_options(),pop)
        pop[:len(elite)] = elite
    answer = pop[np.argmin(fitness(pop))]
    return answer


def exact(goal,ga):
    lo,hi = limits_for(goal)
    choices = len(options)
    # 遗传算法的可行解成本作为上界，随后对同一个离散模型验证最优性。
    from scipy.sparse import vstack
    augmented = vstack([A,csr_matrix(lcc[None,:])])
    cons = LinearConstraint(augmented,np.r_[np.ones(len(s)),lo,-np.inf],
                             np.r_[np.ones(len(s)),hi,lcc[ga].sum()+1e-6])
    res = milp(lcc,integrality=np.ones(choices),bounds=Bounds(0,1),constraints=cons,
               options={"mip_rel_gap":0.0})
    if not res.success:
        raise RuntimeError(res.message)
    chosen = np.flatnonzero(res.x>.5)
    return chosen, res.mip_gap


def verify(plan,goal):
    assert (sid[plan]==stations).all()
    v = values[plan].sum(axis=0)
    lo,hi = limits_for(goal)
    assert (v>=lo-1e-6).all() and (v<=hi+1e-6).all()
    for i,j in enumerate(plan):
        combo = options.at[j,"组合"]
        assert len(combo)<=s.iloc[i]["台数上限"] and set(combo)<=set(valid_products(site_ids[i],s.iloc[i]))


def combo_text(combo):
    return "、".join(f"{k}×{n}" for k,n in sorted(Counter(combo).items())) or "—"


summary,details,scene_rows,hist = [],[],[],[]
for k,(name,goal) in enumerate(goals.items()):
    ga = genetic(goal,20261127+k)
    verify(ga,goal)
    plan, gap = exact(goal,ga)
    verify(plan,goal)
    g = options.iloc[plan].set_axis(s.index)
    served = g["覆盖"].to_numpy()
    fraction = served/demand
    summary.append({"方案":name,"台数":g["台数"].sum(),"投放站数":g["台数"].gt(0).sum(),"充电桩":g["桩数"].sum(),
        "任务覆盖":served.sum(),"覆盖率%":served.sum()/demand.sum()*100,"可替代人力当量":fraction@s["人力"],
        "高危覆盖率%":served[risk_mask].sum()/demand[risk_mask].sum()*100,
        **{c:g[c].sum() for c in ["设备原价","设备净支出","配套","首年运维","年电费","首年支出","次年支出","第三年支出","三年支出","寿命成本"]},
        "遗传算法寿命成本":lcc[ga].sum(),"精确求解间隙%":100*gap})
    counts = Counter(pid for combo in g["组合"] for pid in combo)
    for pid,n in sorted(counts.items()):
        hist.append([name,pid,p.at[pid,"型号"],p.at[pid,"机型"],n,net[pid]*n])
    t = s[["场景","名称","风险","时段","台数上限","需求"]].join(g[["班数","台数","桩数","覆盖","设备净支出","配套","首年运维","年电费","首年支出","寿命成本"]])
    t.insert(0,"方案",name)
    t["产品及数量"] = g["组合"].map(combo_text)
    t["缺口"] = demand-served
    t["可替代人力当量"] = fraction*s["人力"]
    details.append(t.reset_index())
    for c,total,possible,taken in zip(scenes,scene_demand,by_scene@capacity,by_scene@served):
        scene_rows.append([name,c,total,possible,taken,100*taken/total,max(0,target*total-taken)])

summary = pd.DataFrame(summary).set_index("方案")
details = pd.concat(details,ignore_index=True)
coverage = pd.DataFrame(scene_rows,columns=["方案","场景","需求","供给上界","覆盖","覆盖率%","距85%缺口"])
mix = pd.DataFrame(hist,columns=["方案","产品编号","型号","机型","台数","设备净支出"])
details.to_csv(OUT/"三套方案逐站采购.csv",index=False,encoding="utf-8-sig",float_format="%.6f")
print(summary.T.to_string())
print(mix.to_string(index=False))
print(coverage.to_string(index=False))


def table(frame):
    return '<div class="scroll">'+frame.to_html(border=0,float_format=lambda v:f"{v:,.2f}")+'</div>'


rec = summary.loc["最高覆盖"]
middle = summary.loc["均衡覆盖"]
station_table = details[details["方案"]=="最高覆盖"].set_index("站点编号")
zero = station_table.loc[station_table["覆盖"]==0,["场景","名称","需求","产品及数量"]]
scene_table = coverage.pivot(index="场景",columns="方案",values="覆盖率%").reindex(index=scenes,columns=summary.index)
scene_table.insert(0,"总任务量",scene_demand)
scene_table["距85%最低缺口"] = np.maximum(0,target*scene_demand-by_scene@capacity)
mix_table = mix.pivot_table(index=["产品编号","型号","机型"],columns="方案",values="台数",fill_value=0).reindex(columns=summary.index)
cash = summary[["设备原价","设备净支出","配套","首年运维","年电费","首年支出","次年支出","第三年支出","三年支出","寿命成本"]].T
alg = summary[["遗传算法寿命成本","寿命成本","精确求解间隙%"]].copy()
alg["精修节省"] = alg["遗传算法寿命成本"]-alg["寿命成本"]
headline = ["台数","投放站数","充电桩","任务覆盖","覆盖率%","可替代人力当量","高危覆盖率%","首年支出","寿命成本"]
html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题三｜采购与站点分配</title><style>
*{{box-sizing:border-box}}body{{max-width:1180px;margin:auto;padding:38px 28px 64px;font:16px/1.75 system-ui,sans-serif;color:#1e393e}}
h1{{font-size:32px;line-height:1.3}}h2{{margin:35px 0 12px;border-bottom:2px solid #29717b;font-size:22px}}p{{max-width:94ch}}
.scroll{{overflow:auto;margin:12px 0 22px}}table{{border-collapse:collapse;font-size:14px;white-space:nowrap;font-variant-numeric:tabular-nums}}
td,th{{padding:5px 9px;border-bottom:1px solid #d6e2e1;text-align:right}}thead{{background:#e8f1f0}}tbody th{{text-align:left}}
summary{{cursor:pointer;font-weight:650}}small{{color:#52656c}}code{{font-size:15px}}.note{{border-left:4px solid #a05f28;padding:4px 16px;background:#fcf6ed}}
</style><main><h1>问题三｜机器人采购与站点分配</h1>
<p><b>推荐最高覆盖方案。</b>在优先保障可服务的高危任务、让每个有合格机型的站点都完成日常任务的偏好下，购置 {int(rec['台数'])} 台、部署 {int(rec['投放站数'])} 站，日均服务 {rec['任务覆盖']:,.0f} 单。首年财政支出 {rec['首年支出']:,.2f} 万元，低于 2027 年 4,000 万元额度；六年成本现值为 {rec['寿命成本']:,.2f} 万元。</p>
<p class="note">严格执行七类场景全部达到 85% 时，原题数据无可行解。政务接待站 S030 的 28 单/日无机型满足要求，其余三站合计 115 单/日，最多覆盖 115/143＝80.42%。达到 85% 需要 121.55 单/日，差 6.55 单/日。下列方案均以明确记录这项例外为前提；若 85% 是不可豁免的验收线，则三套方案均不能通过验收。</p>
<h2>三套采购方案</h2>{table(summary[headline])}
<p>三个方案分别要求全市覆盖至少 90%、95%、或达到可行上界。共同要求其他六类场景达到 85%、政务达到自身上界、R4/R5 尽可能全部覆盖。在各自要求下最小化六年成本。人力按本站已覆盖任务比例线性折算为全职当量，用于比较方案。高危站总计 {demand[risk_mask].sum()} 单/日，可服务其中 {risk_ceiling:.0f} 单/日。</p>
<p>最高覆盖方案比均衡覆盖方案多 {int(rec['台数']-middle['台数'])} 台，首年多支出 {rec['首年支出']-middle['首年支出']:.2f} 万元，六年成本现值增加 {rec['寿命成本']-middle['寿命成本']:.2f} 万元，日均多覆盖 {rec['任务覆盖']-middle['任务覆盖']:.1f} 单。均衡方案尚未部署的可服务站点为 {'、'.join(details.loc[(details['方案']=='均衡覆盖') & (details['台数']==0) & details['站点编号'].isin(s.index[capacity>0]),'站点编号'])}。设备型号限于第二问已核验可用的候选，选型时保留前 15 名以外的合格机型比价。</p>
<h2>买哪些型号，放在哪些站点</h2>{table(mix_table.astype(int))}
<p>各场景覆盖率按站点实际完成单量汇总，单位为 %。最右列单位为单/日，表示在本候选库下距 85% 的不可消除缺口。</p>{table(scene_table)}
<details open><summary>推荐方案 · 全部 36 个站点</summary>{table(station_table[['场景','名称','风险','班数','需求','产品及数量','台数','桩数','覆盖','缺口','首年支出']])}</details>
<p>上表保留了无机型可投的站点。S008 要求定位误差 ≤1.00 mm，可接受类别的最佳值为 1.13 mm；S030 仅接受双足人形，要求续航 ≥6 h、误差 ≤0.60 mm，而该类最长续航 4.8 h、最佳误差 1.13 mm。因此两站的 50 单/日构成全市供给缺口。另两套方案的逐站数据也已收入 CSV。</p>
<h2>资金与寿命成本（万元）</h2>{table(cash)}
<p>第三问按 2027 年初一次性购置和投运估算。设备按当年补贴抵扣计净采购支出，单台补贴为 min(单价×30%, 25 万元)；配套含每个启用站 8 万元改造费及每桩 6 万元。各站每 3 台配 1 桩，不足 3 台向上取整，跨站不合桩。首年支出包括设备净采购、站点配套、质保减免后的运维和运行电费。第二、三年按继续运行且不追加采购核算。</p>
<p>预算约束为当年设备净支出 ≤3,400 万元，2027 年运维 ≤900 万元，各年支出 ≤4,000 万元，三年合计 ≤12,000 万元。最高覆盖方案设备净支出余量 {net_limit-rec['设备净支出']:.2f} 万元，首年总额余量 {year_budget-rec['首年支出']:.2f} 万元。1.2 亿元是三年专项上限，六年成本现值单独核算。</p>
<p>六年成本现值＝初始设备净支出＋配套费＋六年运维和电费现值−残值现值。运维以购入首年为起点每年上涨 3%，质保内市方承担 50%；按全年 365 日、双班 2.2 kWh/台·日、单班减半、电价 0.75 元/kWh 计电费，金额除以 10,000 换算万元。折现率 5%，经常支出按年末发生，残值取第六年末设备补贴前原价的 10%。</p>
<h2>整数模型与求解</h2>
<p>令 xᵢⱼ 为站点 i 采购产品 j 的非负整数台数，nᵢ＝Σⱼxᵢⱼ。仅允许通过机型、五项性能门槛、适配分阈值、IP、防爆及 TRL 核验的组合取正值；nᵢ ≤ 本站台数上限。附件未给出逐型号功能安全证书及所有 R4 站点的可燃气体标记，按已知字段筛选，并对化工装置区保守要求防爆；落地时仍须核实适用认证。服务时段仅给时间窗，24 小时站按双班，其他按服务窗口 ≤12 小时为单班、&gt;12 小时为双班处理，跨午夜按实际时长。这一分界为本次假设。</p>
<p>站点有效完成量 qᵢ＝min(站点需求 dᵢ, 0.92 × 班次系数 × Σⱼ标称日产能ⱼ × xᵢⱼ)，单班系数 0.5、双班系数 1。双班标称日产能已包含两班；完成量按需求封顶，富余不能跨站抵扣。各场景覆盖率＝Σqᵢ/Σdᵢ，缺口保留真实需求作分母。站点上界来自遍历全部合格组合。用 ε 约束法设置全市覆盖下限与场景均衡要求，将 R4/R5 的覆盖量固定在可达上界后，求各覆盖档的最低成本。</p>
<p>按站枚举 0 到台数上限的整数多重组合，允许同站混配型号；在本站预算、成本及覆盖上均无优势的组合可剪枝。一个染色体含 36 个基因，每个基因选择本站的一种配置。每个覆盖档分别运行遗传算法：种群 120、240 代、4 元锦标赛、均匀交叉、每基因变异率 1.4/36、保留 8 个精英。再以可行解成本为上界，把每站选一项写成 0-1 混合整数规划，用 HiGHS 分支定界精修并校验。下表的最优性仅对应本页列明的假设和各档约束。</p>{table(alg)}
<p><small>来源为赛题 Excel 附件 1 至 4。此页和配套程序为 AI 辅助复算材料。工具为 OpenAI Codex（GPT-6 系列，使用日期 2026-11-27，具体版本发布日期未提供）。</small></p></main></html>"""
(OUT/"问题三分析.html").write_text(html,encoding="utf-8")
