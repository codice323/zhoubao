# 本程序及代码是在 AI 工具辅助下完成的。
# OpenAI Codex，GPT-6 系列；使用日期 2026-11-27，具体版本发布日期未提供。
"""python 复算.py [附件.xlsx]。依赖 numpy、pandas、scipy、openpyxl。"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

out = Path(__file__).resolve().parent
book = Path(sys.argv[1]) if len(sys.argv)>1 else next(out.parent.glob("统一赛题*/附件.xlsx"))
p,s,m,h = [pd.read_excel(book,sheet_name=i,header=1,index_col=0) for i in range(1,5)]
p.columns = ["型号","机型","厂商","自由度","负载","速度","续航","误差","感知","泛化","IP","防爆","单价","运维","质保","TRL","产能"]
s.columns = ["场景","名称","区域","经度","纬度","面积","需求","人力","风险","人工成本","时段","台数",
             "负载","自由度","续航","误差","泛化","适配","机型"]
cfg = pd.read_excel(book,sheet_name=5,header=1).set_index("参数名称")["取值"]
years = (2027,2028,2029)
prices = np.array([cfg[f"技术降价系数（{y}）"] for y in years])
subsidies = np.array([cfg[f"{y} 年购置补贴比例"] for y in years])
budgets = np.array([cfg[f"{y} 年度额度"] for y in years])
wages = np.array([1,cfg["人力成本上涨系数（2028）"],cfg["人力成本上涨系数（2029）"]])
rate,life = cfg["折现率"],int(cfg["设备设计寿命"])
disc = (1+rate)**(-np.arange(1,life+1,dtype=float))
op = np.array([p["运维"]*cfg["运维成本年增系数"]**a*(1-cfg["质保期内运维减免"]*(p["质保"]>a)) for a in range(life)])
I,J=len(s),len(p)
site_cap = s["台数"].to_numpy(dtype=int)
scene = s["场景"].to_numpy()
risky = s["风险"].isin(["R4","R5"]).to_numpy()
def shift(window):
    if window=="24小时":
        return 2
    def minute(t):
        a,b=map(int,t.split(":"))
        return 60*a+b
    a,b=window.split("-")
    return 1 if (minute(b)-minute(a))%1440<=720 else 2

shifts = s["时段"].map(shift).to_numpy()
unitcap = np.outer(shifts/2,p["产能"])*cfg["单台日均产能折算系数"]
energy = shifts/2*cfg["单台日均能耗"]*365*cfg["电价"]/10000
eligible = np.zeros((I,J),dtype=bool)
for i,(sid,r) in enumerate(s.iterrows()):
    req = h.loc[r["风险"]]
    ip = req["最低防护等级"]
    ok = (p["机型"].isin(r["机型"].split(";")) & (p["误差"]<=r["误差"]) & (m.loc[sid,p.index]>=r["适配"])
          & (p["IP"].str[2]>=ip[2]) & (p["IP"].str[3]>=ip[3]) & (p["TRL"]>=req["最低TRL"]))
    for k in ["负载","自由度","续航","泛化"]:
        ok &= p[k]>=r[k]
    if req["是否强制防爆认证"].startswith("是") or (r["风险"]=="R4" and "化工" in r["名称"]):
        ok &= p["防爆"].eq("是")
    eligible[i] = ok
assert (I,J)==(36,48) and eligible.any(axis=0).sum()==13
pile_size = int(cfg["充电桩配比"])
pile_price,site_fee = cfg["充电桩单价"],cfg["单站配套改造费"]
qbase = s["需求"].to_numpy()


@dataclass(frozen=True)
class Case:
    name: str
    price: float = 1
    tech: bool = True
    taper: bool = True
    load: tuple = (1,1,1)


def inputs(case):
    gross = (prices if case.tech else np.ones(3))[:,None]*p["单价"].to_numpy()*case.price
    sub = (subsidies if case.taper else np.repeat(subsidies[0],3))[:,None]
    net = gross-np.minimum(gross*sub,cfg["单台补贴上限"])
    demand = np.outer(case.load,qbase)
    return gross,net,demand


def ceilings(demand,lock,locked_years):
    best = np.where(eligible,unitcap,0).max(axis=1)
    upper = np.minimum(demand,site_cap*best)
    if lock is not None:
        old = lock[:locked_years].sum(axis=0)
        possible = (old*unitcap).sum(axis=1)+(site_cap-old.sum(axis=1))*best
        upper[locked_years:] = np.minimum(demand[locked_years:],possible)
    return upper


def optimize(case,lock=None,immediate=False,locked_years=2):
    gross,net,demand=inputs(case)
    ceiling=ceilings(demand,lock,locked_years)
    bounds_lo,bounds_hi,ints,obj=[],[],[],[]
    def var(low,up,integer=0,cost=0):
        k=len(obj)
        bounds_lo.append(low); bounds_hi.append(up); ints.append(integer); obj.append(cost)
        return k
    # 同站按可用产能及全年度成本删除严格被支配的型号；在役型号始终保留。
    allowed=eligible.copy()
    for i in range(I):
        js=np.flatnonzero(eligible[i])
        clipped=np.minimum(unitcap[i],demand[:,i].max())
        properties=np.column_stack([-clipped,net.T,(net-gross*cfg["残值率"]*disc[-1]+disc@op).T,op.T])
        for j in js:
            if lock is not None and lock[:locked_years,i,j].sum()>0:
                continue
            diff=properties[js]-properties[j]
            if ((diff<=1e-10).all(axis=1)&(diff< -1e-10).any(axis=1)).any():
                allowed[i,j]=False
    rows=[]; lower=[]; upper=[]
    def row(values,lo=-np.inf,hi=np.inf):
        rows.append(values); lower.append(lo); upper.append(hi)
    x={}; q={}; act={}; pile={}
    cash=[{} for _ in years]; purchases=[{} for _ in years]; om0={}
    for t in range(3):
        d0=(1+rate)**-t
        for i in range(I):
            js=np.flatnonzero(allowed[i])
            for j in js:
                low,up=0,site_cap[i]
                if lock is not None and t<locked_years:
                    low=up=lock[t,i,j]
                cost=net[t,j]+disc@(op[:,j]+energy[i])-gross[t,j]*cfg["残值率"]*disc[-1]
                k=x[t,i,j]=var(low,up,1,cost*d0)
                purchases[t][k]=net[t,j]
                for u in range(t,3):
                    cash[u][k]=(net[t,j] if t==u else 0)+op[u-t,j]+energy[i]
                if t==0:
                    om0[k]=op[0,j]
            # 累计配套。目标和预算只收本年相对上年的增量。
            future_discount=(1+rate)**-(t+1) if t<2 else 0
            a=act[t,i]=var(0,1,1,site_fee*(d0-future_discount))
            b=pile[t,i]=var(0,int(np.ceil(site_cap[i]/pile_size)),1,pile_price*(d0-future_discount))
            for k,c in ((a,site_fee),(b,pile_price)):
                cash[t][k]=c
                if t<2:
                    cash[t+1][k]=-c
            v=q[t,i]=var(0,demand[t,i])
            stock={x[u,i,j]:1 for j in js for u in range(t+1)}
            row({**stock,a:-site_cap[i]},hi=0)
            row({**stock,a:-1},lo=0)
            row({**stock,b:-pile_size},lo=1-pile_size,hi=0)
            row({**{x[u,i,j]:-unitcap[i,j] for j in js for u in range(t+1)},v:1},hi=0)
        for c in pd.unique(scene):
            ii=np.flatnonzero(scene==c)
            floor=min(cfg["场景覆盖率下限"]*demand[t,ii].sum(),ceiling[t,ii].sum())
            row({q[t,i]:1 for i in ii},lo=floor)
        row({q[t,i]:1 for i in np.flatnonzero(risky)},lo=ceiling[t,risky].sum())
        floor=[.90*demand[t].sum(),.95*demand[t].sum(),ceiling[t].sum()][t]
        if immediate and t==0:
            floor=ceiling[t].sum()
        row({q[t,i]:1 for i in range(I)},lo=floor)
        row(cash[t],hi=budgets[t])
        row(purchases[t],hi=budgets[t]*cfg["采购支出占比上限"])
    row(om0,hi=cfg["首年运维预算上限"])
    row({k:sum(a.get(k,0) for a in cash) for k in set().union(*cash)},hi=cfg["专项资金总额"])
    rr,cc,dd=[],[],[]
    for idx,items in enumerate(rows):
        for k,v in items.items():
            rr.append(idx); cc.append(k); dd.append(v)
    mat=csr_matrix((dd,(rr,cc)),shape=(len(rows),len(obj)))
    res=milp(obj,integrality=ints,bounds=Bounds(bounds_lo,bounds_hi),
             constraints=LinearConstraint(mat,lower,upper),options={"mip_rel_gap":1e-5,"time_limit":90})
    if not res.success:
        raise RuntimeError(f"{case.name}：{res.message}")
    plan=np.zeros((3,I,J),dtype=int)
    for (t,i,j),k in x.items():
        plan[t,i,j]=round(res.x[k])
    _,summary,_,_=ledger(case,plan)
    assert abs(summary["寿命成本现值"]-res.fun)<.01
    if lock is not None:
        assert np.array_equal(plan[:locked_years],lock[:locked_years])
    actual=np.minimum(demand,(plan.cumsum(axis=0)*unitcap).sum(axis=2))
    for t in range(3):
        assert actual[t,risky].sum()>=ceiling[t,risky].sum()-1e-6
        for c in pd.unique(scene):
            idx=scene==c
            assert actual[t,idx].sum()>=min(cfg["场景覆盖率下限"]*demand[t,idx].sum(),ceiling[t,idx].sum())-1e-6
    assert (actual.sum(axis=1)>=[.90*demand[0].sum()-1e-6,.95*demand[1].sum()-1e-6,ceiling[2].sum()-1e-6]).all()
    return plan,res.mip_gap


def ledger(case,plan):
    gross,net,demand=inputs(case)
    stocks=np.cumsum(plan,axis=0); totals=stocks.sum(axis=2)
    assert (plan>=0).all() and not plan[:,~eligible].any() and (totals<=site_cap).all()
    served=np.minimum(demand,(stocks*unitcap).sum(axis=2))
    cash=[]; details=[]; scenes=[]
    previous_infra=np.zeros(I); pv=0; labor_pv=0
    for t,y in enumerate(years):
        additions=plan[t]
        qty=additions.sum(axis=0)
        sites=totals[t]>0; chargers=(totals[t]+pile_size-1)//pile_size
        current_infra=sites*site_fee+chargers*pile_price
        infra=current_infra-previous_infra; previous_infra=current_infra
        buy=net[t]@qty
        maint=sum(plan[u].sum(axis=0)@op[t-u] for u in range(t+1))
        power=totals[t]@energy
        annual=buy+infra.sum()+maint+power
        assert annual<=budgets[t]+1e-5 and buy<=budgets[t]*cfg["采购支出占比上限"]+1e-5
        assert t or maint<=cfg["首年运维预算上限"]+1e-5
        pv+=(buy+infra.sum()+disc@op@qty+disc.sum()*(additions.sum(axis=1)@energy)
             -gross[t]@qty*cfg["残值率"]*disc[-1])/(1+rate)**t
        labor=(served[t]/demand[t])@(s["人工成本"].to_numpy()*wages[t])
        labor_pv+=labor/(1+rate)**(t+1)
        cash.append([case.name,y,qty.sum(),totals[t].sum(),sites.sum(),chargers.sum(),demand[t].sum(),served[t].sum(),
                     100*served[t].sum()/demand[t].sum(),gross[t]@qty-buy,buy,infra.sum(),maint,power,annual,labor])
        for i, sid in enumerate(s.index):
            def mix(counts):
                return "、".join(f"{p.index[j]}×{counts[j]}" for j in np.flatnonzero(counts)) or "—"
            details.append([case.name,y,sid,scene[i],s.iloc[i]["名称"],mix(additions[i]),mix(stocks[t,i]),
                            additions[i].sum(),totals[t,i],chargers[i],demand[t,i],served[t,i],demand[t,i]-served[t,i]])
        for c in pd.unique(scene):
            mask=scene==c
            ds=demand[t,mask].sum(); qs=served[t,mask].sum()
            scenes.append([case.name,y,c,ds,qs,qs/ds*100,max(0,.85*ds-qs)])
    annual=pd.DataFrame(cash,columns=["情景","年度","新增台数","在役台数","投放站数","累计充电桩","需求","覆盖单/日",
                 "覆盖率%","补贴","设备净支出","新增配套","运维","电费","年度支出","人工成本替代上界"])
    assert annual["年度支出"].sum()<=cfg["专项资金总额"]+.00001
    summary={"情景":case.name,"逐年新增":"/".join(annual["新增台数"].astype(str)),"期末台数":totals[-1].sum(),
        "期末覆盖率%":annual.iloc[-1]["覆盖率%"],"期末缺口":(demand[-1]-served[-1]).sum(),
        "三年支出":annual["年度支出"].sum(),"寿命成本现值":pv,"三年人工替代额现值上界":labor_pv}
    return annual,summary,details,scenes


base=Case("基准滚动")
cases=[base,Case("单价−15%",.85),Case("单价+15%",1.15),
       Case("无技术降价",tech=False),Case("补贴维持30%",taper=False),
       Case("价格补贴均冻结",tech=False,taper=False),
       Case("任务量−20%",load=(.8,)*3),Case("任务量+20%",load=(1.2,)*3),
       Case("2029任务−20%",load=(1,1,.8)),Case("2029任务+20%",load=(1,1,1.2))]
outputs=[]; detail_rows=[]; scene_rows=[]; ledgers=[]; plans={}
for case in cases:
    lock=plans[base.name] if case.name.startswith("2029") else None
    a,gap=optimize(case,lock)
    plans[case.name]=a
    annual,result,rows,scenes=ledger(case,a)
    result["求解间隙%"]=100*gap
    outputs.append(result); detail_rows.extend(rows); scene_rows.extend(scenes); ledgers.append(annual)
    print(result,flush=True)
annual=pd.concat(ledgers,ignore_index=True)
summary=pd.DataFrame(outputs).set_index("情景")
summary["较基准支出变化"]=summary["三年支出"]-summary.at[base.name,"三年支出"]
details=pd.DataFrame(detail_rows,columns=["情景","年度","站点","场景","名称","新增型号及台数","在役型号及台数","新增台数",
    "在役台数","累计充电桩","需求","已覆盖","缺口"])
scene_stats=pd.DataFrame(scene_rows,columns=["情景","年度","场景","需求","已覆盖","覆盖率%","距85%缺口"])
annual.to_csv(out/"年度采购与预算.csv",index=False,float_format="%.6f",encoding="utf-8-sig")
details.to_csv(out/"逐站滚动采购.csv",index=False,float_format="%.6f",encoding="utf-8-sig")

def table(frame):
    return '<div class="scroll">'+frame.to_html(border=0,float_format=lambda v:f"{v:,.2f}")+"</div>"


def change_sites(name,y):
    d=details[(details["情景"]==name)&(details["年度"]==y)&(details["新增台数"]>0)]
    return "、".join(d["站点"]+"（"+d["新增型号及台数"]+"）")


baseline=annual[annual["情景"]==base.name].set_index("年度")
im_plan,_=optimize(base,immediate=True)
im,im_summary,_,_=ledger(base,im_plan)
saved=im_summary["三年支出"]-summary.at[base.name,"三年支出"]
lost=(im["覆盖单/日"].sum()-baseline["覆盖单/日"].sum())*365
procurement_compare=pd.DataFrame([
    ["滚动采购",summary.at[base.name,"逐年新增"],summary.at[base.name,"期末台数"],
     baseline["覆盖单/日"].sum()*365,summary.at[base.name,"三年支出"],summary.at[base.name,"寿命成本现值"]],
    ["2027一次购齐",im_summary["逐年新增"],im_summary["期末台数"],
     im["覆盖单/日"].sum()*365,im_summary["三年支出"],im_summary["寿命成本现值"]],
],columns=["策略","逐年新增","期末台数","三年累计服务量","三年财政支出","寿命成本现值"]).set_index("策略")
view=["逐年新增","期末台数","期末覆盖率%","期末缺口","三年支出","较基准支出变化","寿命成本现值"]
fixed_checks=[]
for case in cases:
    _,res,_,_=ledger(case,plans[base.name])
    fixed_checks.append([case.name,res["期末覆盖率%"],summary.at[case.name,"期末覆盖率%"],res["期末缺口"],summary.at[case.name,"期末缺口"]])
fixed=pd.DataFrame(fixed_checks,columns=["情景","照原清单覆盖率%","重算覆盖率%","照原清单缺口","重算缺口"]).set_index("情景")

# 打包离线静态图，不需要网络字体或脚本。
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fonts={f.name for f in font_manager.fontManager.ttflist}
for family in ["Hiragino Sans GB","PingFang SC","Microsoft YaHei","Noto Sans CJK SC","SimHei"]:
    if family in fonts:
        plt.rcParams["font.family"]=family
        break
fig,ax=plt.subplots(figsize=(8.6,3.6),layout="constrained")
plt.rcParams["axes.unicode_minus"]=False
impact=summary.loc[[c.name for c in cases[1:8]],"较基准支出变化"].sort_values()
bars=ax.barh([k.replace("−","-") for k in impact.index],impact,color=["#367c75" if v<0 else "#ab6234" for v in impact])
ax.axvline(0,color="#6d7779",linewidth=.8)
for bar,v in zip(bars,impact):
    ax.text(v+(8 if v>=0 else -8),bar.get_y()+bar.get_height()/2,f"{v:+.1f}",va="center",ha="left" if v>=0 else "right")
ax.set_xlim(impact.min()-100,impact.max()+100)
ax.set_xlabel("三年财政支出相对基准的变化（万元）")
ax.spines[["top","right","left"]].set_visible(False)
pic=io.BytesIO()
fig.savefig(pic,format="png",dpi=150,bbox_inches="tight"); plt.close(fig)
chart=base64.b64encode(pic.getvalue()).decode()
first=plans[base.name][0].sum(axis=0)
buy2027="、".join(f"{p.index[j]} × {first[j]}" for j in np.flatnonzero(first))
scenario_names=[c.name for c in cases[1:8]]
branch_names=["2029任务−20%",base.name,"2029任务+20%"]
r014=p.index.get_loc("R014")
gross,net,_=inputs(base)
subcaps=cfg["单台补贴上限"]
fiscal_gap=summary.at["任务量+20%","期末缺口"]
locked_gap=summary.at["2029任务+20%","期末缺口"]
upper_labor_diff=im_summary["三年人工替代额现值上界"]-summary.at[base.name,"三年人工替代额现值上界"]
proof=baseline[["新增台数","在役台数","投放站数","累计充电桩","覆盖单/日","覆盖率%","设备净支出","新增配套","运维","电费","年度支出"]]
scene_view=scene_stats[scene_stats["情景"]==base.name].pivot(index="场景",columns="年度",values="覆盖率%")
base_demand=inputs(base)[2]
base_ceiling=ceilings(base_demand,None,2)
reachability=[]
for c in pd.unique(scene):
    idx=scene==c
    ds=base_demand[0,idx].sum(); upper=base_ceiling[0,idx].sum()
    reachability.append([c,ds,upper,100*upper/ds,100*min(cfg["场景覆盖率下限"],upper/ds),
                         max(0,cfg["场景覆盖率下限"]*ds-upper)])
reachability=pd.DataFrame(reachability,columns=["场景","任务量","物理上界","物理上界%","有效考核线%","距85%不可达缺口"]).set_index("场景")
html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题四｜灵敏度分析与滚动采购</title><style>
*{{box-sizing:border-box}}body{{max-width:1180px;margin:auto;padding:32px 28px 64px;color:#203a40;font:16px/1.75 system-ui,sans-serif}}h1{{font-size:32px;line-height:1.3}}
h2{{margin:34px 0 10px;border-bottom:2px solid #356f72;font-size:22px}}h3{{font-size:18px}}p,li{{max-width:95ch}}li{{margin-bottom:8px}}
.scroll{{overflow:auto;margin:12px 0 22px}}table{{border-collapse:collapse;font-size:14px;white-space:nowrap;font-variant-numeric:tabular-nums}}
td,th{{padding:5px 9px;border-bottom:1px solid #dce5e4;text-align:right}}thead{{background:#e8f1f0}}tbody th{{text-align:left}}
img{{max-width:100%;height:auto}}summary{{cursor:pointer;font-weight:650}}small{{color:#51676a}}
</style><main><h1>问题四｜灵敏度分析与滚动采购</h1>
<p>基准情景下，近期两年采购 {int(first.sum())} 台和 {int(baseline.loc[2028,'新增台数'])} 台，远期再购 {int(baseline.loc[2029,'新增台数'])} 台。期末覆盖 {baseline.loc[2029,'覆盖单/日']:,.0f} 单/日。与第三问一次购齐相比，分期减少三年财政支出 {saved:.2f} 万元，代价是前两年累计少完成约 {lost:,.0f} 单。若要求现有可服务站点从 2027 年起即全部覆盖，一次购齐方案仍适用。</p>
<h2>近期两年与远期一年怎样投放</h2>{table(proof)}<p>本表金额均为万元。班次、站点台数上限、可核验的产品门槛与第三问一致。预算每年按 4,000 万元、设备净采购上限 3,400 万元核对，不依赖结转。选型范围涵盖全部 48 款，13 款有可匹配站点。</p>
<ul><li><b>2027 年</b>购置 {buy2027}。优先完成 R4/R5 在已知候选条件下可交付的全部任务。</li>
<li><b>2028 年</b>核对已到货机型、站点实绩和报价后，将 2027 年实际购置量锁定，重算剩余两年。基准预测下的新增为 {change_sites(base.name,2028)}。</li>
<li><b>2029 年</b>只调整新增部分。若任务量与基准一致，新增为 {change_sites(base.name,2029)}。三年的逐站在役量、配桩和缺口均见 CSV。</li></ul>
<details><summary>展开七类场景逐年覆盖率（%）</summary>{table(scene_view)}</details>
<h3>可达覆盖率与有效考核线</h3>{table(reachability)}
<p>附件把 85% 场景覆盖率列为软约束。对场景 k，先根据合格机型、班次和站点台数上限计算物理可达率 ρₖmax，再统一设置有效考核线 min(85%,ρₖmax)。因此政务接待不是人工删除约束，而是按照同一公式取其物理上界 80.42%；若项目验收坚持所有场景均不得低于 85%，则当前产品库无可行方案。</p>
<h2>四类因素对预算与部署的影响</h2>
<p>下表假定情景在 2027 年决策时已知，分别重排全部三年的采购。单价扰动对各年报价统一生效，技术系数在报价上连乘；任务量按每个站点同比例变化。无技术降价固定系数为 1，补贴维持 30% 仍受单台 {subcaps:.0f} 万元上限。其余参数一次只改变表中指定的因素。“价格补贴均冻结”同时冻结技术系数和补贴率。缺口单位为单/日。</p>
{table(summary.loc[[base.name,*scenario_names],view])}
<img src="data:image/png;base64,{chart}" alt="不同预知情景下三年支出与基准的差值，单位万元">
<p>单价上浮与下浮造成的支出变化并不对称，因为高价产品的补贴会触及单台上限。取消技术降价时，采购提前至 {summary.at['无技术降价','逐年新增']} 台。以主力 R014 为例，基准下三年每台补贴后净价为 {net[0,r014]:.2f}、{net[1,r014]:.2f}、{net[2,r014]:.2f} 万元；从 2028 到 2029，补贴退坡足以抵消技术降价。决策还要计入当年是否需要该台产能和资金的时间价值。</p>
<p>任务量增加 20% 后，即使提前知道，S035 科普教育站单班最多部署两台，2 × 170 × 0.92 × 0.5＝156.4 单/日，比其 168 单/日的需求少 11.6 单。另两处缺口位于 S008 和 S030。任务下降时，全期最少寿命成本方案仅需 {summary.at['任务量−20%','期末台数']} 台，说明需求变化会改变设备数量。</p>
<h2>2029 年预测变动后的处理</h2>
<p>锁定基准方案的 2027、2028 年实际批次，共 {baseline.loc[2028,'在役台数']} 台，假设设备继续留在原站运行。若 2029 年全市各站任务量降至基准的 80% 或上升至 120%，只重算最后一年。对照列另按原清单照常购买，直接用新需求检验覆盖率。</p>
{table(summary.loc[branch_names,view].join(fixed['照原清单覆盖率%']))}
<p>需求下降时新增 {change_sites('2029任务−20%',2029)}。需求上升时新增 {change_sites('2029任务+20%',2029)}。若高增长从一开始即已明确，S035 能部署两台高产能机型；沿用基准前两年的 2 台 R014，最多服务 147.2 单/日，并已占满站点上限。相同的期末总台数下，晚获知增长会新增 {locked_gap-fiscal_gap:.1f} 单/日缺口。</p>
<h2>与第三问和赛事政策的衔接</h2>
<p>基准分期三年财政支出 {baseline['年度支出'].sum():,.2f} 万元，一次购齐为 {im_summary['三年支出']:,.2f} 万元；两者各批设备六年寿命成本折算到 2027 年初分别为 {summary.at[base.name,'寿命成本现值']:,.2f} 和 {im_summary['寿命成本现值']:,.2f} 万元。三年累计少完成的任务按每年 365 天估算；只按三年名义现金衡量，提前服务多一单的增量成本约为 {saved*10000/lost:.2f} 元。人工成本按站点覆盖比例和 1、1.08、1.17 的年度系数估算替代额上界；一次购齐较分期的三年替代额现值上界多 {upper_labor_diff:.2f} 万元，是否形成实际节省须以岗位排班和结算记录复核。</p>
{table(procurement_compare)}
<p>滚动方案与一次购齐方案使用相同的终期服务上界和成本口径。前者以较低财政支出换取较慢的前期服务爬坡；后者适合把早期覆盖价值置于资金节约之上。该表把“少花多少钱”和“少服务多少任务”同时列示，避免只按采购成本选择实施节奏。</p>
<ol>
<li><b>按站点设技术补缺项目。</b>工业巡检 S008 的定位误差上限为 1.00 mm，政务接待 S030 同时要求误差 ≤0.60 mm 和续航 ≥6 h。对新机型组织实测，达标后再更新可行清单。核验期间安排人工接续，政务的 115/143＝80.42% 单列，并报告距 85% 的 6.55 单/日。</li>
<li><b>为增长敏感站点预留选型余地。</b>S035 的两台 R014 合计有效产能为 147.2 单/日。若签约前的持续需求预测超过此值，应优先比价更高产能机型；若预测超过 156.4 单/日，则须评估延长服务班次、调整站点上限或人工承接。</li>
<li><b>以补贴后净价安排采购批次。</b>对每个报价逐台计算技术降价与 25 万元补贴上限，财政部门在年度下单前核对补贴资格和额度。R014 的 2028 年净价低于 2029 年，对确定需要的批次应结合服务收益比价。</li>
<li><b>把远期批量设为可调整数量。</b>近期 38 台落实后，2029 年基准需新增 3 台，需求上下浮动 20% 时对应 2 台或 6 台。复核各站最近一段时间的日均有效任务量、机型和可用台数后确定最终数量。</li>
<li><b>把高危任务和安全核验列入采购验收。</b>高危站用实际交付单量作考核，R5 核对防爆证书，R4 核查现场可燃气体条件、功能安全和失联停机措施。维修、停机或认证缺失时保留人工接管。</li>
<li><b>年度预算同时列设备、运维和配套。</b>基准 2029 年设备净采购 {baseline.loc[2029,'设备净支出']:.2f} 万元，同年运维已达 {baseline.loc[2029,'运维']:.2f} 万元。按批次追踪质保到期、运维涨幅、充电桩和站点改造，避免只以设备报价核算专项支出。</li>
<li><b>用实际替岗收益选择实施速度。</b>对比每单提前服务的约 {saved*10000/lost:.2f} 元增量成本与可核实的人工、等待及服务价值；未实际撤减或转用的岗位不计财政节约。</li></ol>
<h2>复算口径</h2>
<p>以站点、型号、采购年为索引设置新增整数变量，在役量为历年新增累计，不提前报废或跨站调拨。用逐站任务量与班次折算产能中的较小值计算交付。各年先把可服务的高危任务覆盖到物理上界；每类场景满足 qₖt/dₖt≥min(85%,ρₖtmax)。基准设全市 2027 年至少 90%、2028 年至少 95%、2029 年达到物理上界，形成逐年提高的服务爬坡路径；前两项是用于比较滚动采购的分析设定，并通过一次购齐方案作对照。</p>
<p>在服务约束下，以三批设备的六年财政寿命成本现值最小为目标，用混合整数规划联合决定三年采购。第 t 年每台净价＝原始报价×价格扰动×技术系数−min(当年补贴率×当年报价, 25)；寿命按购置年起算，首年起按质保内 50% 和运维逐年上涨 3% 核算，残值取购置时补贴前报价的 10%。折现率为 5%，设备与配套于年初支付，运维、电费和寿命终值按年末计。购置价扰动不改变附件单独列明的运维报价。单班与双班有效产能分别为标称值的 0.46 和 0.92，24 小时按双班，其他服务窗以 12 小时分界。按每站每 3 台配 1 桩向上取整，设施只按增量建设。任一年度设备净购置不超额度的 85%；首年运维不超 900 万元；三年总财政支出不超 12,000 万元。所有已列情景的求解相对间隙均低于 0.0011%。</p>
<p><small>数据来自赛题 Excel 的产品库、站点需求、适配矩阵、危险约束及政策参数。使用 OpenAI Codex（GPT-6 系列，2026-11-27，具体版本发布日期未提供）辅助数据复算。分析假设、逐站结果和程序一并提供。</small></p></main></html>"""
(out/"问题四分析.html").write_text(html,encoding="utf-8")
