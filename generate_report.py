#!/usr/bin/env python3
"""
Generate comprehensive PDF report on LLM Agent Evaluation Infrastructure
Incorporates all research notes, data analysis, and charts
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors

# Ensure output directory exists
os.makedirs('output/files/reports', exist_ok=True)

# Create document
doc = SimpleDocTemplate(
    "output/files/reports/llm_agent_evaluation_report.pdf",
    pagesize=letter,
    rightMargin=0.5*inch,
    leftMargin=0.5*inch,
    topMargin=0.6*inch,
    bottomMargin=0.6*inch
)

# Get styles and create custom styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name='CustomBody',
    parent=styles['Normal'],
    fontSize=10,
    leading=13,
    alignment=TA_JUSTIFY
))
styles.add(ParagraphStyle(
    name='CustomHeading1',
    parent=styles['Heading1'],
    fontSize=16,
    leading=18,
    textColor=colors.HexColor('#1a1a1a'),
    spaceAfter=12,
    alignment=TA_LEFT
))
styles.add(ParagraphStyle(
    name='CustomHeading2',
    parent=styles['Heading2'],
    fontSize=13,
    leading=15,
    textColor=colors.HexColor('#333333'),
    spaceAfter=8,
    alignment=TA_LEFT
))
styles.add(ParagraphStyle(
    name='ChartCaption',
    parent=styles['Normal'],
    fontSize=9,
    leading=11,
    textColor=colors.HexColor('#555555'),
    alignment=TA_CENTER,
    spaceAfter=6
))

story = []

# Title and Date
title_style = styles['Title']
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("LLM Agent Evaluation Infrastructure", styles['CustomHeading1']))
story.append(Paragraph("Comprehensive Market Analysis, Frameworks, and Benchmarks", styles['Heading2']))
story.append(Spacer(1, 0.05*inch))
story.append(Paragraph(f"<i>Report Date: {datetime.now().strftime('%B %d, %Y')}</i>", styles['Normal']))
story.append(Spacer(1, 0.2*inch))

# Executive Summary
story.append(Paragraph("1. Executive Summary", styles['CustomHeading1']))
exec_summary = """
The LLM agent evaluation landscape is experiencing explosive growth with global market valuations projected to reach <b>$52.62 billion by 2030</b> from $5.43 billion in 2024 (46.3% CAGR). Enterprise adoption has accelerated dramatically, with <b>79% of organizations</b> implementing AI agents and <b>57.3%</b> running agents in production environments.

Key highlights from 2024-2025 research:
<br/><b>Market Growth:</b> AI agents market expanding from $5.25B (2024) to projected $52.62B (2030); enterprise LLM market reaching $8.8B (2025)
<br/><b>Enterprise Adoption:</b> 72% of enterprise projects now involve multi-agent architectures (up from 23% in 2024); 88% of enterprises report regular AI use
<br/><b>ROI Performance:</b> Average $3.50-6.00 ROI per dollar invested; high-performing organizations achieving 9.3x returns
<br/><b>Framework Ecosystem:</b> LangGraph running in 400+ production companies; CrewAI serving 150+ enterprise customers with 100,000+ daily executions
<br/><b>Standardization:</b> Model Context Protocol (MCP) achieving 80x download growth (100K to 8M downloads in 5 months); rapid cross-vendor adoption
<br/><b>Evaluation Focus:</b> 89% of organizations have implemented observability for agents, while 52% have adopted evaluation frameworks
"""
story.append(Paragraph(exec_summary, styles['CustomBody']))
story.append(Spacer(1, 0.2*inch))

# Section 2: Evaluation Frameworks Landscape
story.append(Paragraph("2. Evaluation Frameworks Landscape", styles['CustomHeading1']))

framework_intro = """
The evaluation framework market has consolidated around specialized tools serving different organizational needs. Leading platforms balance cost efficiency, framework integration, and evaluation depth. Current landscape shows strong segmentation: LangChain ecosystem solutions (LangSmith), open-source alternatives (Langfuse, RAGAS), vendor-neutral platforms (Arize, Braintrust), and specialized research tools (DeepEval).
"""
story.append(Paragraph(framework_intro, styles['CustomBody']))
story.append(Spacer(1, 0.1*inch))

# Framework comparison table
framework_data = [
    ['Framework', 'Type', 'Pricing', 'Key Strengths'],
    ['LangSmith', 'Ecosystem', '$$$', 'Deep LangChain integration, mature platform'],
    ['Langfuse', 'Open-Source', 'Free/\nSelf-hosted', 'Best core features, data control, stable APIs'],
    ['Braintrust', 'Commercial', '$199/mo', 'CI/CD integration, unified PM/eng workflow'],
    ['Arize AX', 'Enterprise', 'Custom', 'Online evals, OpenTelemetry native, 100x cheaper storage'],
    ['DeepEval', 'Research', 'Open-source', '40+ research-backed metrics, component-level evals'],
    ['RAGAS', 'Research', 'Open-source', 'RAG-specific, reference-free, 70% adoption'],
]

framework_table = Table(framework_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1.8*inch])
framework_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
]))

story.append(framework_table)
story.append(Spacer(1, 0.15*inch))

# Add Framework comparison chart
if os.path.exists("output/files/charts/05_framework_comparison.png"):
    try:
        img = Image("output/files/charts/05_framework_comparison.png", width=6.5*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("Figure 1: Framework Feature Comparison Matrix", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load framework chart: {e}")

# Key metrics section
story.append(Paragraph("Key Evaluation Adoption Metrics:", styles['CustomHeading2']))
metrics_text = """
<b>Production Deployment:</b> 57.3% of organizations have agents in production (State of Agent Engineering 2025)<br/>
<b>Observability Implementation:</b> 89% have implemented observability for agents (above evaluation adoption)<br/>
<b>Evaluation Adoption:</b> 52% have adopted evaluation frameworks; 32% cite quality as top barrier to production<br/>
<b>Evaluation Methods:</b> 59.8% use human review for nuanced decisions; 53.3% use LLM-as-judge approaches<br/>
<b>Cost Considerations:</b> 10x cost advantage for online evaluations (Arize vs. LangSmith for 1-year data retention)
"""
story.append(Paragraph(metrics_text, styles['CustomBody']))
story.append(Spacer(1, 0.15*inch))

# Section 3: Benchmark Suites & Performance Data
story.append(PageBreak())
story.append(Paragraph("3. Benchmark Suites & Performance Data", styles['CustomHeading1']))

bench_intro = """
Comprehensive benchmark ecosystems have emerged across multiple domains, measuring agent capabilities from basic tool use to complex autonomous task completion. Leading benchmarks include AgentBench (8 diverse environments), WebArena (812 e-commerce/web tasks), SWE-Bench (real GitHub issues), OSWorld (369 OS/GUI tasks), and GAIA (450 reasoning questions). Performance data shows rapid improvement trajectories and significant model specialization.
"""
story.append(Paragraph(bench_intro, styles['CustomBody']))
story.append(Spacer(1, 0.1*inch))

# Benchmark performance table
bench_data = [
    ['Benchmark', 'Task Count', 'Domain', 'Top Model Performance', 'Key Metric'],
    ['AgentBench', '8 environments', 'Multi-domain', '29 models evaluated', 'Long-term reasoning'],
    ['WebArena', '812 tasks', 'E-commerce/Web', 'Gemini 2.5 Pro: 54.8%', 'Functional correctness'],
    ['SWE-Bench Verified', 'Real issues', 'Code/Engineering', 'Claude 4.5: 77.2%', 'Issue resolution'],
    ['OSWorld', '369 tasks', 'OS/GUI', 'Claude 4.5: 61.4%', 'GUI grounding'],
    ['GAIA', '450 questions', 'General reasoning', 'GPT-4 plugins: 15%', 'Reasoning depth'],
    ['BFCL', '2,000+ examples', 'Function calling', 'GPT-5: Top ranked', 'Call accuracy'],
]

bench_table = Table(bench_data, colWidths=[1.1*inch, 0.95*inch, 1.1*inch, 1.4*inch, 1.35*inch])
bench_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightcyan]),
]))

story.append(bench_table)
story.append(Spacer(1, 0.15*inch))

# Performance trends
story.append(Paragraph("Performance Improvement Trajectories:", styles['CustomHeading2']))
trends_text = """
<b>WebArena (2-Year Progress):</b> 14% success rate (2022) → 60% success rate (2024) = 46 percentage point improvement<br/>
<b>OSWorld (4-Month Acceleration):</b> 42.2% (Claude 4.5 initial) → 61.4% (with improvements) = 19.2 percentage point gain<br/>
<b>Software Engineering:</b> Claude 4.5 Sonnet leads at 77.2% on SWE-bench Verified; first model to exceed 60% on Terminal-Bench<br/>
<b>Reasoning Performance (2025):</b> AIME 2025: Claude Opus 4 reaches 90%; GPQA Diamond: Gemini 3 Pro leads at 91.9%<br/>
<b>Rapid Capability Leap:</b> 2024 agents handled tasks under 30 minutes; 2025 state-of-the-art handles multi-hour complex tasks
"""
story.append(Paragraph(trends_text, styles['CustomBody']))
story.append(Spacer(1, 0.12*inch))

# Add benchmark performance charts
if os.path.exists("output/files/charts/03_benchmark_performance.png"):
    try:
        img = Image("output/files/charts/03_benchmark_performance.png", width=6.5*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("Figure 2: Benchmark Performance Across Models", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load benchmark chart: {e}")

# Section 4: Market Analysis & Adoption Trends
story.append(PageBreak())
story.append(Paragraph("4. Market Analysis & Adoption Trends", styles['CustomHeading1']))

market_intro = """
The LLM agent market is entering a critical growth phase with enterprise adoption accelerating rapidly across all major industries. Market forces include standardization around protocols (MCP, A2A), multi-agent architecture convergence, and production deployment maturation. Enterprise adoption shows a "two-speed" pattern with high-automation sectors deploying at 25% while low-automation sectors remain near 0%, creating significant competitive divergence.
"""
story.append(Paragraph(market_intro, styles['CustomBody']))
story.append(Spacer(1, 0.1*inch))

# Market size metrics
story.append(Paragraph("Market Valuation & Growth Metrics:", styles['CustomHeading2']))
market_metrics = """
<b>Global Agentic AI Market:</b> $5.43B (2024) → $52.62B (2030), 46.3% CAGR<br/>
<b>Enterprise LLM Market:</b> $8.8B (2025) → $55.60B (2032), 30% CAGR<br/>
<b>Generative AI Spending:</b> $37B (2025) vs. $11.5B (2024) - 3.2x increase<br/>
<b>AI Agent Startup Funding:</b> $3.8B (2024) vs. $1.3B (2023) - 192% increase<br/>
<b>Edge AI Hardware:</b> $26.14B (2025) → $58.90B (2030), 17.6% CAGR<br/>
<b>Financial Services Agents:</b> Projected $190.33B by 2030, 30.6% CAGR
"""
story.append(Paragraph(market_metrics, styles['CustomBody']))
story.append(Spacer(1, 0.12*inch))

# Add market growth chart
if os.path.exists("output/files/charts/01_market_growth_trends.png"):
    try:
        img = Image("output/files/charts/01_market_growth_trends.png", width=6.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Paragraph("Figure 3: Agentic AI Market Growth Projections (2024-2030)", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load market growth chart: {e}")

# Enterprise adoption metrics
story.append(Paragraph("Enterprise Adoption Metrics:", styles['CustomHeading2']))
adoption_text = """
<b>Overall Adoption:</b> 79% of organizations have implemented AI agents; 88% report regular AI use<br/>
<b>Production Deployment:</b> 51% of respondents using agents in production today<br/>
<b>Multi-Agent Adoption:</b> 72% of enterprise projects involve multi-agent architectures (up from 23% in 2024)<br/>
<b>Scaling Phase:</b> 23% of organizations actively scaling agentic AI systems; 39% experimenting<br/>
<b>Industry Leaders:</b> Telecommunications (95%), Banking (92%), Healthcare (79%), Manufacturing (77%)<br/>
<b>Fastest Growing:</b> Insurance (8% to 34% - 325% YoY), Legal (14% to 26% - 86% YoY)<br/>
<b>Future Projections:</b> Gartner predicts 33% of enterprise software will include agentic AI by 2028 (33-fold increase)
"""
story.append(Paragraph(adoption_text, styles['CustomBody']))
story.append(Spacer(1, 0.1*inch))

# Add adoption charts
if os.path.exists("output/files/charts/02_adoption_by_industry.png"):
    try:
        img = Image("output/files/charts/02_adoption_by_industry.png", width=6.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Paragraph("Figure 4: AI Agent Adoption by Industry (2025)", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load adoption chart: {e}")

# ROI and productivity
story.append(Paragraph("Return on Investment & Productivity Gains:", styles['CustomHeading2']))
roi_text = """
<b>Overall ROI:</b> $3.50-6.00 return per dollar invested; high-performers achieving 9.3x returns<br/>
<b>Customer Service:</b> $3.50 ROI (14% resolution increase, 9% time reduction); Klarna example: $40M profit improvement<br/>
<b>Code Generation:</b> 126% faster task completion with GitHub Copilot; 55% overall task speed increase<br/>
<b>Financial Services:</b> 250-500% first-year ROI; 171% average mature ROI; 20-40% time savings on routine tasks<br/>
<b>Content Generation:</b> 95% cost reduction (CPG blog example); 50x faster delivery; 20% marketing ROI increase<br/>
<b>Legal Research:</b> 60% reduction in research hours with improved accuracy<br/>
<b>HR Operations:</b> 65% efficiency gain in onboarding and hiring
"""
story.append(Paragraph(roi_text, styles['CustomBody']))
story.append(Spacer(1, 0.1*inch))

# Add ROI chart
if os.path.exists("output/files/charts/03_roi_productivity_gains.png"):
    try:
        img = Image("output/files/charts/03_roi_productivity_gains.png", width=6.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Paragraph("Figure 5: ROI and Productivity Gains by Use Case", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load ROI chart: {e}")

# Section 5: Emerging Technologies & Key Findings
story.append(PageBreak())
story.append(Paragraph("5. Emerging Technologies & Key Findings", styles['CustomHeading1']))

# Framework ecosystem
story.append(Paragraph("LLM Agent Framework Ecosystem:", styles['CustomHeading2']))
framework_text = """
<b>LangGraph:</b> 400+ production deployments (LinkedIn, Uber); fastest framework with lowest latency<br/>
<b>CrewAI:</b> Series A funding ($18M); 150+ enterprise customers; 100,000+ daily agent executions; $3.2M revenue<br/>
<b>AutoGen (Microsoft):</b> 167,000+ GitHub stars; conversational multi-agent workflows; group chat capabilities<br/>
<b>OpenAI Swarm:</b> Lightweight coordination; similar performance to CrewAI; gaining adoption<br/>
<b>LangChain:</b> Most widely used entry point; highest latency among major frameworks; mature ecosystem<br/>
<b>Consolidation Trend:</b> Framework landscape consolidated dramatically (2023-2024); LangGraph/CrewAI emerge as clear leaders
"""
story.append(Paragraph(framework_text, styles['CustomBody']))
story.append(Spacer(1, 0.12*inch))

# Standardization efforts
story.append(Paragraph("Standardization Breakthroughs:", styles['CustomHeading2']))
standards_text = """
<b>Model Context Protocol (MCP):</b> Announced November 2024; achieved 80x growth (100K to 8M downloads in 5 months); 5,800+ servers available; 300+ clients<br/>
<b>Cross-Vendor Adoption Speed:</b> Achieved major adoption in 1 year vs. 5+ years for historical protocols (OAuth 2.0, OpenAPI)<br/>
<b>Vendor Support:</b> Unanimous adoption: Anthropic, OpenAI (March 2025), Google DeepMind (April 2025), Microsoft<br/>
<b>Enterprise Adoption:</b> Block, Bloomberg, Amazon, 100+ Fortune 500 companies deploying MCP<br/>
<b>Linux Foundation Governance:</b> MCP donated to Agentic AI Foundation (December 2025)<br/>
<b>Agent-to-Agent (A2A) Protocol:</b> Google introduced April 2025; Linux Foundation project launched June 2025; enables secure inter-agent communication
"""
story.append(Paragraph(standards_text, styles['CustomBody']))
story.append(Spacer(1, 0.12*inch))

# Add evaluation methods and adoption barriers charts
if os.path.exists("output/files/charts/04_evaluation_methods.png"):
    try:
        img = Image("output/files/charts/04_evaluation_methods.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 6: Evaluation Methods Adoption", styles['ChartCaption']))
        story.append(Spacer(1, 0.05*inch))
    except Exception as e:
        print(f"Could not load evaluation methods chart: {e}")

if os.path.exists("output/files/charts/06_adoption_barriers.png"):
    try:
        img = Image("output/files/charts/06_adoption_barriers.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 7: Top Adoption Barriers", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load barriers chart: {e}")

# Reasoning models
story.append(Paragraph("Reasoning Models & Breakthroughs:", styles['CustomHeading2']))
reasoning_text = """
<b>Reasoning Model Era:</b> OpenAI o1 (Dec 2024) pioneered approach; DeepSeek-R1 (Jan 2025) provided open-weight alternative<br/>
<b>DeepSeek-R1 Performance:</b> AIME 2024: 15.6% initial → 77.9% after training → 86.7% with self-consistency<br/>
<b>Industry Response:</b> Every major model provider released reasoning variant by 2025 (Anthropic, Google, Microsoft)<br/>
<b>Performance Impact:</b> Claude 3.7 Sonnet and o3 reach ~82% on Polyglot benchmark (225 coding problems)<br/>
<b>Inference Cost:</b> 1.6x output token increase with reasoning; significant budget implications for cost-sensitive deployments<br/>
<b>Research Availability:</b> Pure RL approach enables reasoning learning without human-labeled trajectories
"""
story.append(Paragraph(reasoning_text, styles['CustomBody']))
story.append(Spacer(1, 0.12*inch))

# Add performance and hallucination charts
if os.path.exists("output/files/charts/04_performance_metrics.png"):
    try:
        img = Image("output/files/charts/04_performance_metrics.png", width=6.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Paragraph("Figure 8: Agent Performance Metrics Comparison", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load performance chart: {e}")

# Section 6: Key Findings and Recommendations
story.append(PageBreak())
story.append(Paragraph("6. Key Findings & Recommendations", styles['CustomHeading1']))

# Critical findings
story.append(Paragraph("Critical Findings:", styles['CustomHeading2']))

findings_list = """
<b>1. Two-Speed Enterprise Landscape:</b> High-automation sectors deploying agents at 25% adoption while low-automation firms near 0%, creating significant competitive divergence. This indicates winners will emerge quickly in 2025-2026.<br/><br/>

<b>2. Production Deployment Gap Remains Critical:</b> 70% of organizations run pilots but less than 20% achieve full-scale production deployment. This is the primary blocker preventing ROI realization and explains only ~25% of AI initiatives deliver expected returns.<br/><br/>

<b>3. Evaluation is Below Observability Adoption:</b> 89% have implemented observability while only 52% have evaluation frameworks. Quality is cited as top barrier (32%) to production deployment, indicating evaluation tools/practices remain a critical gap.<br/><br/>

<b>4. Standardization Momentum Unprecedented:</b> MCP achieved 80x growth and cross-vendor adoption in 1 year vs. 5+ years for historical protocols. This convergence around MCP, A2A, and function calling standards will dramatically accelerate vendor ecosystem maturation.<br/><br/>

<b>5. Multi-Agent Systems Now Standard Practice:</b> 72% of enterprise projects involve multi-agent architectures (up from 23% in 2024). Single-agent applications are rapidly becoming legacy patterns.<br/><br/>

<b>6. Significant Performance Generalization Gaps:</b> SWE-bench Verified (45%) to Pro (23%) shows 22-point drop suggesting potential benchmark contamination/memorization. True reasoning capabilities may be lower than headline scores suggest.<br/><br/>

<b>7. Framework Consolidation Complete:</b> LangGraph, CrewAI, AutoGen emerged as clear leaders. Organizations adopting these frameworks gain significant advantage over those building custom solutions.<br/><br/>

<b>8. Cost Per Interaction Becoming Critical Variable:</b> Token efficiency varies 1.5-4x between models despite similar per-token pricing. Reasoning models cost 1.6x baseline. Model selection now heavily impacts operating margins.
"""

story.append(Paragraph(findings_list, styles['CustomBody']))
story.append(Spacer(1, 0.15*inch))

# Adoption challenges
story.append(Paragraph("Primary Adoption Challenges:", styles['CustomHeading2']))
challenges_text = """
<b>Security & Compliance (35%):</b> Organizations cite cybersecurity and compliance as primary obstacle. Requires enterprise-grade monitoring, RBAC, and audit trails.<br/>
<b>Legacy System Integration (60%):</b> Difficulty integrating agentic AI with existing enterprise infrastructure remains major blocker. API modernization prerequisites.<br/>
<b>Unclear Business Value (62%):</b> Most enterprises lack clear starting point for agent deployment. Pilot projects often lack production path.<br/>
<b>Employee Resistance (87%):</b> Internal resistance from employees remains highest barrier. Fear of job displacement and unfamiliarity with collaborative AI workflows.<br/>
<b>Project Cancellation Risk:</b> Gartner predicts 40%+ of agentic AI projects will be canceled by 2027 due to escalating costs and unclear business value.
"""
story.append(Paragraph(challenges_text, styles['CustomBody']))
story.append(Spacer(1, 0.12*inch))

# Add hallucination and adoption challenge charts
if os.path.exists("output/files/charts/05_hallucination_rates.png"):
    try:
        img = Image("output/files/charts/05_hallucination_rates.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 9: Hallucination Rates by Model", styles['ChartCaption']))
        story.append(Spacer(1, 0.05*inch))
    except Exception as e:
        print(f"Could not load hallucination chart: {e}")

if os.path.exists("output/files/charts/06_roi_analysis.png"):
    try:
        img = Image("output/files/charts/06_roi_analysis.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 10: ROI Achievement Rates", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load ROI analysis chart: {e}")

# Strategic Recommendations
story.append(Paragraph("Strategic Recommendations (2025-2026):", styles['CustomHeading2']))
recommendations = """
<b>For Enterprise Leaders:</b> Adopt standardized frameworks (LangGraph/CrewAI) rather than custom builds. Start with high-ROI use cases (customer service, code generation, research) with clear KPIs. Establish production readiness criteria before pilot expansion. Budget for observability and evaluation infrastructure (52-89% adoption gap indicates investment opportunity).<br/><br/>

<b>For Evaluation Infrastructure:</b> Invest in observability platforms first (89% adoption) while developing evaluation methodologies. Use human-in-the-loop (59.8%) combined with LLM-as-judge (53.3%) for production quality gates. Establish baseline benchmarks before deployment for drift detection. Consider cost advantage of online evaluation platforms (100x cheaper storage vs. monolithic solutions).<br/><br/>

<b>For Model Selection:</b> Account for token efficiency (1.5-4x variation) in cost modeling, not just per-token pricing. Evaluate reasoning model ROI - 1.6x token cost trades off for improved complex reasoning. Test on domain-specific benchmarks (not just generic ones) before production deployment. Expect 20-30% performance drops on robustness variants (multilingual, paraphrased).<br/><br/>

<b>For Product Teams:</b> Plan for multi-agent architectures from inception (72% of enterprise projects now standard). Adopt MCP for tool integration (5,800+ servers available). Design for observability from day one. Implement progressive deployment: confined agents → defined scope → measured autonomy. Prepare for 40%+ project cancellation rate by having clear business value articulation.
"""
story.append(Paragraph(recommendations, styles['CustomBody']))
story.append(Spacer(1, 0.15*inch))

# Add remaining charts
if os.path.exists("output/files/charts/07_adoption_barriers.png"):
    try:
        img = Image("output/files/charts/07_adoption_barriers.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 11: Adoption Barriers Detail", styles['ChartCaption']))
        story.append(Spacer(1, 0.05*inch))
    except Exception as e:
        print(f"Could not load detailed barriers chart: {e}")

if os.path.exists("output/files/charts/08_mcp_adoption.png"):
    try:
        img = Image("output/files/charts/08_mcp_adoption.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 12: MCP Adoption Growth", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load MCP chart: {e}")

# 2025-2027 Outlook
story.append(PageBreak())
story.append(Paragraph("7. 2025-2027 Market Outlook", styles['CustomHeading1']))

outlook_text = """
<b>2025 (Current):</b> Multi-agent systems become standard enterprise pattern (72% adoption). MCP standardization drives rapid tool ecosystem expansion. Evaluation infrastructure investment accelerates. First cohort of scaled agent deployments reaches production maturity, proving ROI models. Consolidation around frameworks (LangGraph, CrewAI, AutoGen) accelerates.<br/><br/>

<b>2026 Projections:</b> Task-specific agent adoption reaches 40% (from <5% baseline) per Gartner. Autonomous agents deploy at scale in 50% of enterprises. Cross-platform agent orchestration becomes standard. Danfoss-style real-time decision automation spreads across industries. First major productivity tool disruptions begin (Gartner: 35-year dominance challenged). Project cancellation rate reaches 40%+ due to escalating costs and unclear value.<br/><br/>

<b>2027+ Outlook:</b> Superhuman AI coders threshold (March 2027). Autonomous AI researchers operational. Claude systems achieving autonomous breakthroughs. "Fast takeoff" scenarios discussed by AI researchers become relevant. New vendor landscape emerges as agentic experiences shift value. USD 58B market disruption as winners consolidate, losers exit. Potential GDP "ballooning" from autonomous decision-making at scale.
"""
story.append(Paragraph(outlook_text, styles['CustomBody']))
story.append(Spacer(1, 0.15*inch))

# Add remaining specialty charts
if os.path.exists("output/files/charts/09_framework_adoption.png"):
    try:
        img = Image("output/files/charts/09_framework_adoption.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 13: Framework Ecosystem Adoption", styles['ChartCaption']))
        story.append(Spacer(1, 0.05*inch))
    except Exception as e:
        print(f"Could not load framework adoption chart: {e}")

if os.path.exists("output/files/charts/10_tool_use_improvements.png"):
    try:
        img = Image("output/files/charts/10_tool_use_improvements.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 14: Tool Use Capability Improvements", styles['ChartCaption']))
        story.append(Spacer(1, 0.05*inch))
    except Exception as e:
        print(f"Could not load tool use chart: {e}")

if os.path.exists("output/files/charts/11_reasoning_comparison.png"):
    try:
        img = Image("output/files/charts/11_reasoning_comparison.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 15: Reasoning Model Performance", styles['ChartCaption']))
        story.append(Spacer(1, 0.05*inch))
    except Exception as e:
        print(f"Could not load reasoning chart: {e}")

if os.path.exists("output/files/charts/12_failure_analysis.png"):
    try:
        img = Image("output/files/charts/12_failure_analysis.png", width=3.2*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 16: Agent Failure Mode Analysis", styles['ChartCaption']))
        story.append(Spacer(1, 0.1*inch))
    except Exception as e:
        print(f"Could not load failure analysis chart: {e}")

# Conclusion
story.append(Paragraph("Conclusion", styles['CustomHeading1']))
conclusion_text = """
The LLM agent evaluation infrastructure is entering a critical inflection point. Market growth (46.3% CAGR to $52.62B by 2030), standardization momentum (MCP 80x growth), and enterprise adoption acceleration (79% implementation rate) create significant opportunities and risks. Organizations that address the production deployment gap (70% pilot, <20% scale), adopt standardized frameworks, and invest in evaluation infrastructure will capture disproportionate value. The two-speed enterprise landscape indicates competitive winners will emerge quickly in 2025-2026. Critical success factors include clear business value articulation, quality-first evaluation strategies, legacy system integration planning, and change management for employee adoption barriers (87% resistance). The next 18-24 months will determine market leaders and establish architectural patterns for the agentic AI era.
"""
story.append(Paragraph(conclusion_text, styles['CustomBody']))
story.append(Spacer(1, 0.2*inch))

# Sources
story.append(PageBreak())
story.append(Paragraph("Sources & References", styles['CustomHeading1']))
sources_text = """
<b>Primary Research:</b> State of Agent Engineering 2025 (LangChain), McKinsey State of AI 2025, Gartner Strategic Predictions 2026, Grand View Research LLM Market Report, AI 2027 Report<br/><br/>

<b>Market Analysis:</b> Menlo Ventures 2025 LLM Market Update, Fortune Business Insights Enterprise LLM Market, Deloitte AI Enterprise Adoption Guide, Google Cloud AI Business Trends 2026<br/><br/>

<b>Frameworks & Evaluation:</b> LangChain State of Agent Engineering, Anthropic Bloom Framework, RAGAS Documentation, OpenTelemetry for AI Observability, Arize LLM Evaluation Platforms<br/><br/>

<b>Benchmarks:</b> AgentBench (ICLR 2024), WebArena Benchmark, SWE-bench Official Leaderboard, OSWorld GitHub, GAIA Benchmark, Berkeley Function Calling Leaderboard<br/><br/>

<b>Standardization:</b> Model Context Protocol Specification, Google Developers Blog (Agent2Agent Protocol), Linux Foundation Agentic AI Foundation, OpenAI MCP Integration<br/><br/>

<b>Industry Adoption:</b> PwC AI Survey 2025, Enterprise AI Adoption Trends 2025-2026, AI Agent Statistics 2025-2026 (Multiple sources), Vector Database Growth Reports<br/><br/>

<b>Academic & Research:</b> Evaluation and Benchmarking of LLM Agents Survey (arXiv), DeepSeek-R1 Nature Paper, Meta-Reinforcement Learning Survey, Tool Learning Research (Springer)<br/><br/>

<i>Data collection period: January 2025 - Present. Report incorporates 200+ data points from 100+ authoritative sources covering 2024-2027 timeframe.</i>
"""
story.append(Paragraph(sources_text, styles['CustomBody']))

# Build PDF
try:
    doc.build(story)
    print(f"PDF successfully generated: output/files/reports/llm_agent_evaluation_report.pdf")
except Exception as e:
    print(f"Error generating PDF: {e}")
    raise
