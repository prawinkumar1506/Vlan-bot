import streamlit as st
import pandas as pd
from data_handler import load_data, plot_metric, get_data_summary, get_anomalies, load_holiday_data, get_holiday_summary, get_upcoming_holidays, load_exam_data, get_exam_summary, get_upcoming_exams
import google.generativeai as genai
from PIL import Image
import ast
import json
from datetime import datetime

# --- CONFIG ---
st.set_page_config(
    page_title="Telemetry Chatbot",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .stChatContainer {
        overflow-x: auto;
    }
    pre, code {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📡 Advanced Telemetry Data Analyzer</p>', unsafe_allow_html=True)
st.markdown("*Intelligent network monitoring with AI-powered insights*")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.header("⚙️ Configuration")

    # API Key Input
    st.subheader("🔑 Gemini API Key")
    api_key = st.text_input("Enter your Gemini API Key", type="password", key="api_key")
    st.caption("🔒 Your key is encrypted and used only for this session.")

    st.divider()

    # Model Selection
    st.subheader("🤖 Model Settings")
    model_name = st.selectbox(
        "Select Gemini Model",
        ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Choose the AI model for analysis"
    )

    temperature = st.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more factual, Higher = more creative"
    )

    st.divider()

    # Display Options
    st.subheader("📊 Display Options")
    show_data_preview = st.checkbox("Show Data Preview", value=False)
    show_statistics = st.checkbox("Show Statistics", value=True)
    enable_anomaly_detection = st.checkbox("Enable Anomaly Detection", value=False)
    show_holidays = st.checkbox("Show Holidays", value=True)

    st.divider()

    # Export Options
    st.subheader("💾 Export")
    if st.button("📥 Download Chat History", use_container_width=True):
        if "chat_history" in st.session_state and st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="Download JSON",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pop("last_metric", None)
        st.session_state.pop("last_vlan", None)
        st.rerun()

# --- MAIN CONTENT ---
if not api_key:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to start analyzing telemetry data.")
    st.info("💡 **Tip:** Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
    st.stop()

# --- INITIALIZE GEMINI ---
try:
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(model_name, generation_config=generation_config)
    st.sidebar.success("✅ API Connected")
except Exception as e:
    st.error(f"❌ Error connecting to Gemini API: {str(e)}")
    st.stop()

# --- LOAD DATA ---
try:
    df = load_data()
    data_summary = get_data_summary(df)

    # Anomaly detection
    anomalies = None
    if enable_anomaly_detection:
        anomalies = get_anomalies(df)

    # Load holiday data
    holiday_df = load_holiday_data()
    holiday_summary = get_holiday_summary(holiday_df)

    # Load exam data
    exam_df = load_exam_data()
    exam_summary = get_exam_summary(exam_df)

except FileNotFoundError as e:
    st.error(f"❌ Error: {str(e)}. Please ensure the file exists in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# --- DATA OVERVIEW ---
if show_statistics:
    st.subheader("📊 Data Overview")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("Total Records", data_summary['total_records'])
    with col_stat2:
        st.metric("VLANs Monitored", data_summary['total_vlans'])
    with col_stat3:
        st.metric("Time Range", data_summary['time_range'])
    with col_stat4:
        st.metric("Avg Input Rate", f"{data_summary['avg_input_rate']:,.0f} bps")

if show_data_preview:
    with st.expander("🔍 View Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=300)

if enable_anomaly_detection and anomalies is not None and not anomalies.empty:
    with st.expander("⚠️ Detected Anomalies", expanded=True):
        st.warning(f"Found {len(anomalies)} potential anomalies in the data")
        st.dataframe(anomalies, use_container_width=True)

# --- HOLIDAY INFORMATION ---
if show_holidays:
    st.subheader("🎉 Holiday Information")

    # Display holiday summary
    col_hol1, col_hol2, col_hol3, col_hol4 = st.columns(4)
    with col_hol1:
        st.metric("Total Holidays", holiday_summary['total_holidays'])
    with col_hol2:
        st.metric("Earliest Holiday", holiday_summary['earliest_holiday'])
    with col_hol3:
        st.metric("Latest Holiday", holiday_summary['latest_holiday'])

    # Show upcoming holidays
    upcoming_holidays = get_upcoming_holidays(holiday_df)
    if not upcoming_holidays.empty:
        with st.expander("📅 Upcoming Holidays", expanded=True):
            st.dataframe(
                upcoming_holidays[['Date', 'Holiday']],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No upcoming holidays in the dataset.")

    # Show holiday distribution
    with st.expander("📊 Holiday Distribution", expanded=False):
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.subheader("By Month")
            for month, count in holiday_summary['holidays_by_month'].items():
                st.write(f"- {month}: {count}")

        with col_dist2:
            st.subheader("By Day of Week")
            for day, count in holiday_summary['holidays_by_day'].items():
                st.write(f"- {day}: {count}")

# --- EXAM INFORMATION ---
st.subheader("📚 Exam Information")

# Display exam summary
col_exam1, col_exam2, col_exam3, col_exam4 = st.columns(4)
with col_exam1:
    st.metric("Total Exams", exam_summary['total_exams'])
with col_exam2:
    st.metric("Earliest Exam", exam_summary['earliest_exam'])
with col_exam3:
    st.metric("Latest Exam", exam_summary['latest_exam'])

# Show upcoming exams
upcoming_exams = get_upcoming_exams(exam_df)
if not upcoming_exams.empty:
    with st.expander("📅 Upcoming Exams", expanded=True):
        st.dataframe(
            upcoming_exams[['Date', 'Start Time', 'End Time', 'Exam']],
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No upcoming exams in the dataset.")

# Show exam distribution
with st.expander("📊 Exam Distribution", expanded=False):
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.subheader("By Type")
        for exam_type, count in exam_summary['exams_by_type'].items():
            st.write(f"- {exam_type}: {count}")

    with col_dist2:
        st.subheader("By Month")
        for month, count in exam_summary['exams_by_month'].items():
            st.write(f"- {month}: {count}")

st.divider()

# --- CHAT LAYOUT ---
col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    st.subheader("💬 AI Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        welcome_msg = {
            "role": "assistant",
            "content": f"""👋 Hello! I'm your telemetry data analyst. I'm currently monitoring **{data_summary['total_vlans']} VLANs** with **{data_summary['total_records']} data points**.

I can help you with:
- 📈 Analyzing traffic patterns
- 🔍 Comparing VLAN performance
- 📊 Identifying peak usage times
- 💡 Detecting anomalies and trends
- 🎉 Providing information about upcoming holidays
- 📚 Providing information about upcoming exams

We have **{holiday_summary['total_holidays']} holidays** and **{exam_summary['total_exams']} exams** in our database.

Try asking: *"Which VLAN has the highest output rate?"* or *"Show me the traffic trends"* or *"What holidays are coming up?"* or *"What exams are scheduled?"*"""
        }
        st.session_state.chat_history.append(welcome_msg)
    
    # Display chat history
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        if st.button("📊 Compare VLANs", use_container_width=True):
            st.session_state.suggested_query = "Compare the performance of all VLANs"
    with col_q2:
        if st.button("🔝 Peak Traffic", use_container_width=True):
            st.session_state.suggested_query = "When is the peak traffic time?"
    with col_q3:
        if st.button("🎉 Holidays", use_container_width=True):
            st.session_state.suggested_query = "What holidays are coming up?"
    with col_q4:
        if st.button("📚 Exams", use_container_width=True):
            st.session_state.suggested_query = "What exams are scheduled?"
    
    # Chat input
    user_input = st.chat_input("Ask about your telemetry data...")
    
    # Handle suggested queries
    if "suggested_query" in st.session_state:
        user_input = st.session_state.suggested_query
        del st.session_state.suggested_query
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show processing indicator
        with st.spinner("🤔 Analyzing data..."):
            # Get available VLANs and metrics
            available_vlans = df['VLAN'].unique().tolist()
            available_metrics = list(df.columns)[2:]
            
            # Enhanced prompt for Gemini
            prompt = f"""
You are an expert network telemetry analyst with deep knowledge of network performance metrics, holiday information, and exam schedules.

DATASET INFORMATION:
- Columns: {list(df.columns)}
- VLANs available: {available_vlans}
- Metrics tracked:
  * Input Rate (bps): Incoming data rate in bits per second
  * Input Packets: Count of incoming packets
  * Output Rate (bps): Outgoing data rate in bits per second
  * Output Packets: Count of outgoing packets
- Total records: {len(df)}
- Time range: {data_summary['time_range']}

HOLIDAY INFORMATION:
- Total holidays: {holiday_summary['total_holidays']}
- Holidays include: {', '.join([h['Holiday'] for h in holiday_summary['holiday_list']])}
- Upcoming holidays: {len(holiday_summary['holiday_list'])} holidays available

EXAM INFORMATION:
- Total exams: {exam_summary['total_exams']}
- Exam types: {', '.join(exam_summary['exams_by_type'].keys())}
- Upcoming exams: {len(exam_summary['exam_list'])} exams available

DATA STATISTICS:
{json.dumps(data_summary['vlan_stats'], indent=2)}

USER QUERY: "{user_input}"

INSTRUCTIONS:
1. Analyze the query and provide a concise, accurate, data-driven answer
2. Use actual statistics from the data provided above
3. If the question is about holidays, use the holiday information
4. If the question is about exams, use the exam information
5. Suggest the most relevant metric to visualize (if applicable)
6. Optionally suggest a specific VLAN (must be from: {available_vlans})
7. If comparing VLANs, leave vlan as null to show all

RESPONSE FORMAT (valid Python dict):
{{"answer": "your detailed analysis with specific numbers", "metric": "exact_metric_name_or_null", "vlan": "VLAN_name_or_null"}}

IMPORTANT:
- VLAN must be exactly from: {available_vlans}
- Metric must be exactly from: {available_metrics} or null if not applicable
- Include specific numbers and insights
- Be professional and analytical
- If the question is about holidays, provide holiday information instead of telemetry data
- If the question is about exams, provide exam information instead of telemetry data
"""
            
            try:
                response = model.generate_content(prompt)
                
                # Parse response
                try:
                    # Try to extract JSON from markdown code blocks
                    response_text = response.text
                    if "```" in response_text:
                        response_text = response_text.split("```")[1]
                        if response_text.startswith("python") or response_text.startswith("json"):
                            response_text = "\n".join(response_text.split("\n")[1:])
                    
                    result = ast.literal_eval(response_text.strip())
                    
                    # Validate the response
                    if not isinstance(result, dict):
                        raise ValueError("Response is not a dictionary")
                    
                    if "answer" not in result:
                        result["answer"] = response.text
                    
                except Exception as parse_error:
                    st.warning(f"⚠️ Response parsing issue: {str(parse_error)}")
                    result = {
                        "answer": response.text,
                        "metric": "Input Rate (bps)",
                        "vlan": None
                    }
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"]
                })
                
                # Store visualization parameters
                st.session_state["last_metric"] = result.get("metric")
                st.session_state["last_vlan"] = result.get("vlan")
                
                # Rerun to update chat display
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })

with col2:
    st.subheader("📈 Interactive Visualization")
    
    if "last_metric" in st.session_state and st.session_state["last_metric"]:
        metric = st.session_state["last_metric"]
        vlan = st.session_state.get("last_vlan")
        
        # Visualization controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            override_metric = st.selectbox(
                "Select Metric",
                list(df.columns)[2:],
                index=list(df.columns)[2:].index(metric) if metric in list(df.columns)[2:] else 0,
                key="metric_select"
            )
        with col_ctrl2:
            vlan_options = ["All VLANs"] + df['VLAN'].unique().tolist()
            current_vlan_idx = 0
            if vlan and vlan in vlan_options:
                current_vlan_idx = vlan_options.index(vlan)
            
            override_vlan = st.selectbox(
                "Select VLAN",
                vlan_options,
                index=current_vlan_idx,
                key="vlan_select"
            )
        
        # Update if user changed selection
        final_metric = override_metric
        final_vlan = None if override_vlan == "All VLANs" else override_vlan
        
        # Generate and display plot
        try:
            with st.spinner("📊 Generating visualization..."):
                buf = plot_metric(df, vlan=final_vlan, metric=final_metric)
                image = Image.open(buf)
                st.image(
                    image,
                    caption=f"{final_metric} over time ({'All VLANs' if not final_vlan else final_vlan})",
                    use_container_width=True
                )
                
                # Download button for plot
                buf.seek(0)
                st.download_button(
                    label="📥 Download Plot",
                    data=buf,
                    file_name=f"telemetry_{final_metric.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"❌ Error generating plot: {str(e)}")
    else:
        st.info("💡 **Get Started:** Ask a question in the chat to generate your first visualization!")
        st.markdown("""
        **Example questions:**
        - "Show me the traffic pattern for VLAN106"
        - "Which VLAN has the highest output rate?"
        - "Compare input rates across all VLANs"
        - "What's the trend in packet transmission?"
        - "What holidays are coming up?"
        - "What exams are scheduled?"
        """)

# --- FOOTER ---
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small></small>
</div>
""", unsafe_allow_html=True)
