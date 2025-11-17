import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Continuous DeFi Insurance Simulator", layout="wide")

st.title("🛡️ Continuous DeFi Insurance Protocol Simulator")

with st.sidebar:
    st.header("⚙️ Market Parameters")
    seed = st.number_input("Random Seed (price generation)", 0, 100, 42)
    base_apr = st.slider("Base Premium APR", 0.05, 0.50, 0.15, 0.01, format="%.2f")
    k_util = st.slider("Utilization Sensitivity (k)", 0.5, 5.0, 2.0, 0.1)
    payout_mult = st.slider("Payout Multiplier", 2.0, 10.0, 5.0, 0.5)
    protocol_fee = st.slider("Protocol Fee", 0.0, 0.20, 0.10, 0.01, format="%.2f")
    lock_days = st.number_input("Lock Period (days)", 1, 30, 7)

    st.header("📊 Risk Function")
    risk_shape = st.selectbox(
        "Risk Function Type",
        [
            "Linear (proportional)",
            "Quadratic (accelerating)",
            "Exponential - Mild (k=10)",
            "Exponential - Moderate (k=15)",
            "Exponential - Aggressive (k=20)",
        ],
        index=4,
    )
    target_price = st.number_input("Target Price", 0.5, 2.0, 1.0, 0.01)

    st.header("👥 Participants")
    st.subheader("Insured Positions")
    n_insured = st.number_input("Number of Insured", 1, 10, 1)
    insured_positions = []
    for i in range(n_insured):
        pos = st.number_input(f"Insured {i+1} Coverage ($)", 10000, 10000000, 100000 * (i + 1), 10000, key=f"ins_{i}")
        insured_positions.append(pos)

    st.subheader("LP Positions")
    n_lps = st.number_input("Number of LPs", 1, 10, 1)
    lp_positions = []
    for i in range(n_lps):
        pos = st.number_input(f"LP {i+1} Liquidity ($)", 10000, 10000000, 500000 * (i + 1), 10000, key=f"lp_{i}")
        lp_positions.append(pos)

    st.header("📈 Simulation")
    sim_days = st.slider("Simulation Days", 10, 365, 100)
    scenario = st.selectbox(
        "Price Scenario",
        [
            "Stable (±2%)",
            "Mild Depeg (drops to 0.96)",
            "Severe Depeg (drops to 0.92)",
            "Extreme Depeg (drops to 0.85)",
            "Volatile (random swings)",
        ],
    )


def get_risk_function(shape_name):
    if "Linear" in shape_name:
        return lambda deviation: deviation
    elif "Quadratic" in shape_name:
        return lambda deviation: deviation**2
    elif "Mild" in shape_name:
        return lambda deviation: 1 - np.exp(-10 * deviation)
    elif "Moderate" in shape_name:
        return lambda deviation: 1 - np.exp(-15 * deviation)
    elif "Aggressive" in shape_name:
        return lambda deviation: 1 - np.exp(-20 * deviation)
    return lambda deviation: deviation


def generate_price_scenario(scenario, days, target, seed):
    hours = days * 24
    np.random.seed(seed)

    if "Stable" in scenario:
        price = target + np.random.randn(hours) * 0.005
        price = np.clip(price, target * 0.98, target * 1.02)
    elif "Mild" in scenario:
        price = np.ones(hours) * target
        price[: int(hours * 0.4)] += np.random.randn(int(hours * 0.4)) * 0.003
        depeg_start = int(hours * 0.4)
        depeg_end = int(hours * 0.75)
        price[depeg_start:depeg_end] = np.linspace(target, 0.96, depeg_end - depeg_start)
        price[depeg_end:] = 0.96 + np.random.randn(hours - depeg_end) * 0.005
    elif "Severe" in scenario:
        price = np.ones(hours) * target
        price[: int(hours * 0.4)] += np.random.randn(int(hours * 0.4)) * 0.003
        depeg_start = int(hours * 0.4)
        depeg_end = int(hours * 0.75)
        price[depeg_start:depeg_end] = np.linspace(target, 0.92, depeg_end - depeg_start)
        price[depeg_end:] = 0.92
    elif "Extreme" in scenario:
        price = np.ones(hours) * target
        price[: int(hours * 0.3)] += np.random.randn(int(hours * 0.3)) * 0.003
        depeg_start = int(hours * 0.3)
        price[depeg_start:] = np.linspace(target, 0.85, hours - depeg_start)
    else:
        price = target + np.cumsum(np.random.randn(hours) * 0.008)
        price = np.clip(price, target * 0.85, target * 1.15)

    return price


def run_simulation(insured_pos, lp_pos, params):
    print("\n" + "=" * 80)
    print("SIMULATION START")
    print("=" * 80)

    total_coverage = sum(insured_pos)
    total_liquidity = sum(lp_pos)
    util = total_coverage / total_liquidity

    print(f"Total Coverage: ${total_coverage:,.0f}")
    print(f"Total Liquidity: ${total_liquidity:,.0f}")
    print(f"Utilization: {util*100:.2f}%")

    util_capped = min(util, 0.99)
    premium_apr = params["base_apr"] * (1 + params["k_util"] * util_capped / (1 - util_capped))

    print(f"Base APR: {params['base_apr']*100:.2f}%")
    print(f"Dynamic Premium APR: {premium_apr*100:.2f}%")
    print(f"Payout Multiplier: {params['payout_mult']}x")
    print(f"Protocol Fee: {params['protocol_fee']*100:.1f}%")

    hours = params["sim_days"] * 24
    t_days = np.linspace(0, params["sim_days"], hours)

    price = generate_price_scenario(params["scenario"], params["sim_days"], params["target"], seed=42)

    risk_func = get_risk_function(params["risk_shape"])
    deviation = np.abs(price / params["target"] - 1)
    f_t = risk_func(deviation)

    print(f"\nRisk f(t) - Min: {f_t.min():.4f}, Max: {f_t.max():.4f}, Mean: {f_t.mean():.4f}")

    dt = 1 / (365 * 24)

    gross_premium_per_hour = premium_apr * total_coverage * dt
    net_premium_per_hour = gross_premium_per_hour * (1 - params["protocol_fee"])

    print(f"\nPer Hour Flows:")
    print(f"  Gross Premium: ${gross_premium_per_hour:.2f}")
    print(f"  Net Premium (after fee): ${net_premium_per_hour:.2f}")

    payout_factor = f_t * params["payout_mult"]
    print(
        f"\nPayout Factor - Min: {payout_factor.min():.4f}, Max: {payout_factor.max():.4f}, Mean: {payout_factor.mean():.4f}"
    )

    net_flow_per_hour = net_premium_per_hour * (payout_factor - 1)

    print(f"\nNet Flow per Hour - Min: ${net_flow_per_hour.min():.2f}, Max: ${net_flow_per_hour.max():.2f}")

    insured_results = []
    for i, cov in enumerate(insured_pos):
        print(f"\n--- Insured {i+1} (Coverage: ${cov:,.0f}) ---")

        premium_paid_per_hour = premium_apr * cov * dt
        cum_premium_paid = np.cumsum(premium_paid_per_hour * np.ones(hours))

        print(f"  Premium per hour: ${premium_paid_per_hour:.4f}")
        print(f"  Total premiums paid: ${cum_premium_paid[-1]:,.2f}")

        flow_received_per_hour = np.zeros(hours)
        flow_received_per_hour[net_flow_per_hour > 0] = net_flow_per_hour[net_flow_per_hour > 0] * (
            cov / total_coverage
        )
        cum_flow_received = np.cumsum(flow_received_per_hour)

        print(f"  Total flow received: ${cum_flow_received[-1]:,.2f}")

        pnl = cum_flow_received - cum_premium_paid
        print(f"  Final PnL: ${pnl[-1]:,.2f}")

        years = t_days / 365
        years = np.maximum(years, 0.01)
        cumulative_apr = (pnl / cov) / years * 100

        instant_pnl_per_hour = flow_received_per_hour - premium_paid_per_hour
        instant_apr = (instant_pnl_per_hour / cov) * (365 * 24) * 100

        print(f"  Final Cumulative APR: {cumulative_apr[-1]:.2f}%")
        print(f"  Final Instant APR: {instant_apr[-1]:.2f}%")

        insured_results.append(
            {
                "id": i + 1,
                "coverage": cov,
                "premium_paid": cum_premium_paid,
                "flow_received": cum_flow_received,
                "pnl": pnl,
                "cumulative_apr": cumulative_apr,
                "instant_apr": instant_apr,
            }
        )

    lp_results = []
    for j, liq in enumerate(lp_pos):
        print(f"\n--- LP {j+1} (Liquidity: ${liq:,.0f}) ---")

        # LP SEMPRE riceve la sua quota di premiums (al netto della fee)
        premium_received_per_hour = net_premium_per_hour * (liq / total_liquidity)
        cum_premium_received = np.cumsum(premium_received_per_hour * np.ones(hours))

        # LP paga quando gli Insured ricevono (quando net_flow > 0)
        payout_paid_per_hour = np.zeros(hours)
        payout_paid_per_hour[net_flow_per_hour > 0] = net_flow_per_hour[net_flow_per_hour > 0] * (liq / total_liquidity)
        cum_payout_paid = np.cumsum(payout_paid_per_hour)

        print(f"  Total premiums received: ${cum_premium_received[-1]:,.2f}")
        print(f"  Total payouts paid: ${cum_payout_paid[-1]:,.2f}")

        pnl = cum_premium_received - cum_payout_paid
        print(f"  Final PnL: ${pnl[-1]:,.2f}")

        years = t_days / 365
        years = np.maximum(years, 0.01)
        cumulative_apr = (pnl / liq) / years * 100

        instant_pnl_per_hour = premium_received_per_hour - payout_paid_per_hour
        instant_apr = (instant_pnl_per_hour / liq) * (365 * 24) * 100

        print(f"  Final Cumulative APR: {cumulative_apr[-1]:.2f}%")
        print(f"  Final Instant APR: {instant_apr[-1]:.2f}%")

        lp_results.append(
            {
                "id": j + 1,
                "liquidity": liq,
                "flow_received": cum_premium_received,
                "flow_paid": cum_payout_paid,
                "pnl": pnl,
                "cumulative_apr": cumulative_apr,
                "instant_apr": instant_apr,
            }
        )

    protocol_fees = np.cumsum(gross_premium_per_hour * params["protocol_fee"] * np.ones(hours))

    print(f"\nProtocol Fees Collected: ${protocol_fees[-1]:,.2f}")

    total_insured_pnl = sum([ins["pnl"][-1] for ins in insured_results])
    total_lp_pnl = sum([lp["pnl"][-1] for lp in lp_results])

    print(f"\n--- TOTALS ---")
    print(f"Total Insured PnL: ${total_insured_pnl:,.2f}")
    print(f"Total LP PnL: ${total_lp_pnl:,.2f}")
    print(f"Protocol Fees: ${protocol_fees[-1]:,.2f}")
    print(f"Sum (should be ~0): ${total_insured_pnl + total_lp_pnl + protocol_fees[-1]:,.2f}")

    print("=" * 80)
    print("SIMULATION END")
    print("=" * 80 + "\n")

    return {
        "t_days": t_days,
        "price": price,
        "f_t": f_t,
        "payout_factor": payout_factor,
        "net_flow": np.cumsum(net_flow_per_hour),
        "premium_apr": premium_apr,
        "utilization": util,
        "insured": insured_results,
        "lps": lp_results,
        "protocol_fees": protocol_fees,
        "total_coverage": total_coverage,
        "total_liquidity": total_liquidity,
    }


if st.sidebar.button("🚀 Run Simulation", type="primary"):
    params = {
        "base_apr": base_apr,
        "k_util": k_util,
        "payout_mult": payout_mult,
        "protocol_fee": protocol_fee,
        "risk_shape": risk_shape,
        "target": target_price,
        "sim_days": sim_days,
        "scenario": scenario,
    }

    results = run_simulation(insured_positions, lp_positions, params)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Coverage", f"${results['total_coverage']:,.0f}")
    with col2:
        st.metric("Total Liquidity", f"${results['total_liquidity']:,.0f}")
    with col3:
        st.metric("Utilization", f"{results['utilization']*100:.1f}%")
    with col4:
        st.metric("Dynamic Premium APR", f"{results['premium_apr']*100:.1f}%")

    breakeven_factor = 1 / (1 - protocol_fee)
    breakeven_ft = breakeven_factor / payout_mult
    st.info(f"📌 **Insured Breakeven:** f(t) must exceed {breakeven_ft:.3f} (factor = {breakeven_factor:.2f})")

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Price", "Risk Function f(t)", "Payout Factor", "Cumulative Net Flow"),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    fig.add_trace(
        go.Scatter(x=results["t_days"], y=results["price"], name="Price", line=dict(color="blue", width=2)),
        row=1,
        col=1,
    )
    fig.add_hline(y=target_price, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

    fig.add_trace(
        go.Scatter(x=results["t_days"], y=results["f_t"], name="f(t)", fill="tozeroy", line=dict(color="red", width=2)),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=results["t_days"], y=results["payout_factor"], name="Payout Factor", line=dict(color="purple", width=2)
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", opacity=0.5, row=3, col=1)
    fig.add_hline(y=breakeven_factor, line_dash="dot", line_color="orange", opacity=0.7, row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=results["t_days"],
            y=results["net_flow"],
            name="Net Flow",
            fill="tozeroy",
            line=dict(color="green", width=2),
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=4, col=1)

    fig.update_xaxes(title_text="Days", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Risk Level", row=2, col=1)
    fig.update_yaxes(title_text="Factor", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative ($)", row=4, col=1)

    fig.update_layout(height=1200, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Insured Participants")

        fig_insured_cum = go.Figure()
        for ins in results["insured"]:
            fig_insured_cum.add_trace(
                go.Scatter(
                    x=results["t_days"],
                    y=ins["cumulative_apr"],
                    name=f"Insured {ins['id']} (${ins['coverage']:,.0f})",
                    line=dict(width=3),
                )
            )
        fig_insured_cum.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_insured_cum.update_layout(
            title="Insured Cumulative APR", xaxis_title="Days", yaxis_title="APR (%)", height=400
        )
        st.plotly_chart(fig_insured_cum, use_container_width=True)

        fig_insured_inst = go.Figure()
        for ins in results["insured"]:
            fig_insured_inst.add_trace(
                go.Scatter(
                    x=results["t_days"],
                    y=ins["instant_apr"],
                    name=f"Insured {ins['id']} (${ins['coverage']:,.0f})",
                    line=dict(width=3),
                )
            )
        fig_insured_inst.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_insured_inst.update_layout(
            title="Insured Instant APR", xaxis_title="Days", yaxis_title="APR (%)", height=400
        )
        st.plotly_chart(fig_insured_inst, use_container_width=True)

        insured_data = []
        for ins in results["insured"]:
            insured_data.append(
                {
                    "ID": ins["id"],
                    "Coverage": f"${ins['coverage']:,.0f}",
                    "Premiums Paid": f"${ins['premium_paid'][-1]:,.0f}",
                    "Payouts Received": f"${ins['flow_received'][-1]:,.0f}",
                    "Final PnL": f"${ins['pnl'][-1]:,.0f}",
                    "Final Cumulative APR": f"{ins['cumulative_apr'][-1]:.2f}%",
                    "Final Instant APR": f"{ins['instant_apr'][-1]:.2f}%",
                }
            )
        st.dataframe(pd.DataFrame(insured_data), use_container_width=True)

    with col2:
        st.subheader("💰 LP Participants")

        fig_lp_cum = go.Figure()
        for lp in results["lps"]:
            fig_lp_cum.add_trace(
                go.Scatter(
                    x=results["t_days"],
                    y=lp["cumulative_apr"],
                    name=f"LP {lp['id']} (${lp['liquidity']:,.0f})",
                    line=dict(width=3),
                )
            )
        fig_lp_cum.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_lp_cum.update_layout(title="LP Cumulative APR", xaxis_title="Days", yaxis_title="APR (%)", height=400)
        st.plotly_chart(fig_lp_cum, use_container_width=True)

        fig_lp_inst = go.Figure()
        for lp in results["lps"]:
            fig_lp_inst.add_trace(
                go.Scatter(
                    x=results["t_days"],
                    y=lp["instant_apr"],
                    name=f"LP {lp['id']} (${lp['liquidity']:,.0f})",
                    line=dict(width=3),
                )
            )
        fig_lp_inst.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_lp_inst.update_layout(title="LP Instant APR", xaxis_title="Days", yaxis_title="APR (%)", height=400)
        st.plotly_chart(fig_lp_inst, use_container_width=True)

        lp_data = []
        for lp in results["lps"]:
            lp_data.append(
                {
                    "ID": lp["id"],
                    "Liquidity": f"${lp['liquidity']:,.0f}",
                    "Premiums Received": f"${lp['flow_received'][-1]:,.0f}",
                    "Payouts Paid": f"${lp['flow_paid'][-1]:,.0f}",
                    "Final PnL": f"${lp['pnl'][-1]:,.0f}",
                    "Final Cumulative APR": f"{lp['cumulative_apr'][-1]:.2f}%",
                    "Final Instant APR": f"{lp['instant_apr'][-1]:.2f}%",
                }
            )
        st.dataframe(pd.DataFrame(lp_data), use_container_width=True)

    st.subheader("📊 Summary Statistics")
    total_insured_pnl = sum([ins["pnl"][-1] for ins in results["insured"]])
    total_lp_pnl = sum([lp["pnl"][-1] for lp in results["lps"]])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Insured PnL", f"${total_insured_pnl:,.0f}")
    with col2:
        st.metric("Total LP PnL", f"${total_lp_pnl:,.0f}")
    with col3:
        st.metric("Protocol Fees Collected", f"${results['protocol_fees'][-1]:,.0f}")

    st.success(
        f"✅ Zero-sum check: Insured PnL + LP PnL + Fees = ${total_insured_pnl + total_lp_pnl + results['protocol_fees'][-1]:,.2f} (should be ≈ $0)"
    )
