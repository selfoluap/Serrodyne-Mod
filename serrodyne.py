import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Modulation Signal Generation Functions ---

def generate_one_segment_waveform(segment, num_points_per_cycle):
    """
    Generates the time and voltage waveform for a single, independent modulation segment
    based on its individual duration and cycle count.
    """
    seg_type = segment.get('type')
    duration_ns = segment.get('duration_ns', 100.0)
    num_cycles = segment.get('num_cycles', 1)
    
    segment_duration_s = duration_ns * 1e-9
    
    if segment_duration_s <= 0:
        return np.array([]), np.array([])
        
    # Calculate the frequency for this specific segment
    base_ramp_frequency = num_cycles / segment_duration_s
    
    num_segment_points = int(num_cycles * num_points_per_cycle)

    if num_segment_points == 0:
        return np.array([]), np.array([])

    t_local = np.linspace(0, segment_duration_s, num_segment_points, endpoint=False)
    V_segment = np.zeros_like(t_local)

    if seg_type == 'ramp':
        slope_direction = segment.get('slope_direction', 'up')
        slope_V_per_cycle = segment.get('slope_V_per_cycle', 0.0)
        voltage_mode = segment.get('voltage_mode', 'positive')

        # Generate a base sawtooth waveform that repeats according to the segment's frequency
        base_sawtooth = np.mod(t_local * base_ramp_frequency, 1)
        
        ramp_shape = (1 - base_sawtooth) if slope_direction == 'down' else base_sawtooth

        if voltage_mode == 'bipolar':
            ramp_shape -= 0.5

        V_segment = slope_V_per_cycle * ramp_shape

    elif seg_type == 'no_wave':
        V_segment = np.zeros_like(t_local)

    return t_local, V_segment


def generate_arbitrary_sequence_modulation(
    sequence_definition,
    V_pi,
    num_points_per_cycle=2**10
):
    """
    Generates a full sequence of voltage and phase modulation by creating a master
    time vector and filling it with independently generated segments. This ensures a
    uniform time step for accurate FFT.
    """
    total_duration_s = sum(seg.get('duration_ns', 0.0) for seg in sequence_definition) * 1e-9
    total_points = sum(int(seg.get('num_cycles', 1) * num_points_per_cycle) for seg in sequence_definition)

    if total_points == 0 or total_duration_s == 0:
        return np.array([0]), np.array([0]), np.array([0]), 0, "Empty Sequence", 0

    t_full = np.linspace(0, total_duration_s, total_points, endpoint=False)
    V_modulating_full = np.zeros(total_points)
    
    current_point_index = 0
    description_parts = []
    max_freq = 0.0

    for segment in sequence_definition:
        num_cycles = segment.get('num_cycles', 1)
        
        # Generate the isolated waveform for this segment
        _t_local, V_segment = generate_one_segment_waveform(segment, num_points_per_cycle)
        
        if V_segment.size == 0:
            continue
            
        # Place the generated segment into the master voltage array
        end_point_index = current_point_index + len(V_segment)
        # Ensure we don't write past the end of the array
        if end_point_index > len(V_modulating_full):
            end_point_index = len(V_modulating_full)
            V_segment = V_segment[:end_point_index - current_point_index]

        V_modulating_full[current_point_index:end_point_index] = V_segment
        current_point_index = end_point_index

        # --- Get metadata for plot titles and scaling ---
        duration_ns = segment.get('duration_ns', 1.0)
        if duration_ns > 0:
            segment_freq = num_cycles / (duration_ns * 1e-9)
            if segment_freq > max_freq:
                max_freq = segment_freq

        seg_type = segment['type']
        if seg_type == 'ramp':
            slope_direction = segment.get('slope_direction', 'up')
            slope_V_per_cycle = segment.get('slope_V_per_cycle', 0.0)
            dir_desc = "UpRamp" if slope_direction == 'up' else "DnRamp"
            description_parts.append(f"{num_cycles}x{dir_desc}({slope_V_per_cycle:.1f}V/cyc)")
        elif seg_type == 'no_wave':
            description_parts.append(f"{num_cycles}xNoWave")
            
    phase_EOM_full = np.pi * (V_modulating_full / V_pi)
    full_description = "Sequence: " + " -> ".join(description_parts)

    return t_full, V_modulating_full, phase_EOM_full, total_duration_s, full_description, max_freq


# --- Main Visualization Function (modified for Streamlit and relative FFT) ---

def arbitrary_sequence_modulation_visualization_streamlit(
    sequence_definition,
    V_pi,
    optical_freq_0,
    f_visual_carrier_proxy=5e6,
    num_points_per_cycle=2**10
):
    """
    Simulates and visualizes an arbitrary sequence of optical modulation.
    """
    t, V_modulating, phase_EOM, total_duration, sequence_description, max_freq = \
        generate_arbitrary_sequence_modulation(
            sequence_definition, V_pi, num_points_per_cycle
        )
    dt = t[1] - t[0] if len(t) > 1 else 0

    # --- Signals Generation ---
    original_sine_wave_proxy = np.sin(2 * np.pi * f_visual_carrier_proxy * t)
    optical_field_original_for_fft = np.ones_like(t)
    optical_field_modulated_for_fft = np.exp(1j * phase_EOM)

    # --- FFT Calculation ---
    def calculate_fft(signal, dt):
        if dt == 0 or len(signal) < 2:
            return np.array([0]), np.array([0])
        Y = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), d=dt)
        Y_shifted = np.fft.fftshift(Y)
        freq_shifted = np.fft.fftshift(freq)
        return freq_shifted, np.abs(Y_shifted)**2

    freq_orig_fft, psd_orig_fft = calculate_fft(optical_field_original_for_fft, dt)
    freq_mod_fft, psd_mod_fft = calculate_fft(optical_field_modulated_for_fft, dt)
    
    # --- Plotting ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)

    # Plot 1
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t * 1e9, original_sine_wave_proxy, color='orange')
    ax0.set_title(f'1. Original Carrier Signal (Visual Proxy, {f_visual_carrier_proxy/1e6:.1f} MHz)\nActual Carrier: {optical_freq_0/1e12:.3f} THz')
    ax0.set_ylabel('Amplitude (a.u.)')
    ax0.set_xlabel('Time (ns)')
    ax0.grid(True)

    # Plot 2
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t * 1e9, V_modulating, color='blue')
    ax1.set_title(f'2. Modulating Voltage Sequence\n{sequence_description}')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_xlabel('Time (ns)')
    ax1.grid(True)
    if len(V_modulating) > 1:
        max_v_mod, min_v_mod = np.max(V_modulating), np.min(V_modulating)
        v_padding = (max_v_mod - min_v_mod) * 0.1 if (max_v_mod - min_v_mod) > 1e-9 else 0.1
        ax1.set_ylim(min_v_mod - v_padding, max_v_mod + v_padding)

    # Plot 3
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t * 1e9, np.degrees(phase_EOM), color='green')
    ax2.set_title(f'3. Modulated Optical Phase Shift ($V_{{\\pi}}$={V_pi}V)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_xlabel('Time (ns)')
    ax2.grid(True)

    # Plot 4
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t * 1e9, optical_field_modulated_for_fft.real, color='purple')
    ax3.set_title('4. Real(Modulated Optical Field Component)')
    ax3.set_ylabel('Amplitude (a.u.)')
    ax3.set_xlabel('Time (ns)')
    ax3.grid(True)

    # Plot 5
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(freq_orig_fft / 1e6, psd_orig_fft, color='darkred')
    ax4.set_title('5. FFT of Original Signal (Relative)')
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Power Spectral Density (a.u.)')
    freq_range = max_freq * 2 if max_freq > 0 else 1e7
    ax4.set_xlim(-freq_range / 1e6, freq_range / 1e6)
    ax4.axvline(0, color='gray', linestyle='--', label='Carrier (Reference)')
    ax4.grid(True)
    ax4.legend()

    # Plot 6
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(freq_mod_fft / 1e6, psd_mod_fft, color='darkgreen')
    ax5.set_title(f'6. FFT of Modulated Signal (Relative to Carrier)')
    ax5.set_xlabel('Frequency (MHz)')
    ax5.set_ylabel('Power Spectral Density (a.u.)')
    ax5.set_xlim(-freq_range / 1e6, freq_range / 1e6)
    
    # Find and show peak frequency, excluding the DC component
    if len(psd_mod_fft) > 0:
        center_index = len(psd_mod_fft) // 2
        # Define a small region around the center to exclude from the peak search
        dc_exclusion_width = 5 
        
        # Create a copy of the PSD and set the DC component to zero to ignore it
        psd_no_dc = np.copy(psd_mod_fft)
        psd_no_dc[center_index - dc_exclusion_width : center_index + dc_exclusion_width] = 0

        # Find the index of the absolute maximum peak
        peak_index = np.argmax(psd_no_dc)
            
        peak_freq = freq_mod_fft[peak_index]
        peak_psd = psd_mod_fft[peak_index]

        # Add a vertical line and text to mark the peak
        ax5.axvline(peak_freq / 1e6, color='magenta', linestyle='--', label=f'Peak: {peak_freq / 1e6:.2f} MHz')
        ax5.text(peak_freq / 1e6, peak_psd, f' {peak_freq / 1e6:.2f} MHz', color='magenta', ha='left', va='bottom')

    ax5.axvline(0, color='blue', linestyle='--', label='Carrier Component')
    ax5.grid(True)
    ax5.legend()

    plt.tight_layout()
    return fig

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="EOM Modulation Simulator")
st.title("Electro-Optic Modulator (EOM) Modulation Simulator")

# --- Global Parameters ---
st.sidebar.header("Global Parameters")
c = 299792458
lambda_0 = st.sidebar.number_input("Optical Wavelength (nm)", value=870.0, min_value=1.0, format="%.1f") * 1e-9
optical_frequency_0_hz = c / lambda_0
st.sidebar.write(f"Calculated Optical Frequency: {optical_frequency_0_hz/1e12:.3f} THz")
V_pi_eom = st.sidebar.number_input("EOM $V_{\\pi}$ (Volts)", value=2.0, min_value=0.1, format="%.1f")
num_points_per_cycle = st.sidebar.number_input("Points per Base Ramp Cycle", value=2**10, min_value=128, max_value=2**12, step=128)

# --- Predefined Sequences ---
st.sidebar.header("Predefined Sequences")

def load_preset_1():
    st.session_state.sequence_definition = [
        {'type': 'no_wave', 'duration_ns': 100.0, 'num_cycles': 3, 'slope_V_per_cycle': None},
        {'type': 'ramp', 'duration_ns': 200.0, 'num_cycles': 6, 'slope_direction': 'down', 'slope_V_per_cycle': 4.0 * V_pi_eom, 'voltage_mode': 'bipolar'},
        {'type': 'no_wave', 'duration_ns': 100.0, 'num_cycles': 3, 'slope_V_per_cycle': None}
    ]

def load_preset_2():
    st.session_state.sequence_definition = [
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'down', 'slope_V_per_cycle': 6.0, 'voltage_mode': 'bipolar'},
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'down', 'slope_V_per_cycle': 1.5 * V_pi_eom, 'voltage_mode': 'bipolar'},
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'up', 'slope_V_per_cycle': 1.5 * V_pi_eom, 'voltage_mode': 'bipolar'},
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'up', 'slope_V_per_cycle': 6.0, 'voltage_mode': 'bipolar'}
    ]

def load_preset_3():
    st.session_state.sequence_definition = [
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'down', 'slope_V_per_cycle': 3.0, 'voltage_mode': 'bipolar'},
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'down', 'slope_V_per_cycle': 1.0, 'voltage_mode': 'bipolar'},
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'up', 'slope_V_per_cycle': 1.0, 'voltage_mode': 'bipolar'},
        {'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 4, 'slope_direction': 'up', 'slope_V_per_cycle': 3.0, 'voltage_mode': 'bipolar'}
    ]

col_preset1, col_preset2, col_preset3 = st.sidebar.columns(3)
with col_preset1:
    st.button("Preset 1", on_click=load_preset_1, use_container_width=True)
with col_preset2:
    st.button("Preset 2", on_click=load_preset_2, use_container_width=True)
with col_preset3:
    st.button("Preset 3", on_click=load_preset_3, use_container_width=True)


# --- Sequence Definition ---
st.sidebar.header("Define Modulation Sequence")

if 'sequence_definition' not in st.session_state:
    st.session_state.sequence_definition = [
        {'type': 'ramp', 'duration_ns': 150.0, 'num_cycles': 3, 'slope_direction': 'down', 'slope_V_per_cycle': 2.0 * V_pi_eom, 'voltage_mode': 'positive'},
        {'type': 'no_wave', 'duration_ns': 100.0, 'num_cycles': 2, 'slope_V_per_cycle': None},
    ]

def add_segment():
    st.session_state.sequence_definition.append({'type': 'ramp', 'duration_ns': 100.0, 'num_cycles': 1, 'slope_direction': 'up', 'slope_V_per_cycle': V_pi_eom, 'voltage_mode': 'positive'})

def remove_segment(index):
    if len(st.session_state.sequence_definition) > 1:
        st.session_state.sequence_definition.pop(index)

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    st.button("Add Segment", on_click=add_segment)

st.sidebar.subheader("Sequence Segments:")
for i, segment in enumerate(st.session_state.sequence_definition):
    with st.sidebar.expander(f"Segment {i+1}", expanded=True):
        segment['type'] = st.selectbox(f"Type##{i}", ['ramp', 'no_wave'], index=0 if segment.get('type') == 'ramp' else 1, key=f"seg_type_{i}")
        
        segment['duration_ns'] = st.number_input(f"Segment Time (ns)##{i}", value=float(segment.get('duration_ns', 100.0)), min_value=1.0, format="%.1f", key=f"seg_duration_{i}")
        segment['num_cycles'] = st.number_input(f"Number of Cycles##{i}", value=segment.get('num_cycles', 1), min_value=1, key=f"seg_cycles_{i}")

        if segment['type'] == 'ramp':
            segment['slope_direction'] = st.selectbox(f"Slope Direction##{i}", ['up', 'down'], index=0 if segment.get('slope_direction', 'up') == 'up' else 1, key=f"seg_slope_{i}")
            initial_slope_value = float(segment.get('slope_V_per_cycle') or V_pi_eom)
            segment['slope_V_per_cycle'] = st.number_input(f"Slope (V/cycle)##{i}", value=initial_slope_value, min_value=0.0, format="%.2f", key=f"seg_slope_V_per_cycle_{i}")
            segment['voltage_mode'] = st.selectbox(f"Voltage Mode##{i}", ['positive', 'bipolar'], format_func=lambda x: f"Positive (0 to Vmax)" if x == 'positive' else f"Bipolar (-0.5Vmax to +0.5Vmax)", index=0 if segment.get('voltage_mode', 'positive') == 'positive' else 1, key=f"seg_voltage_mode_{i}")
        else:
            segment['slope_V_per_cycle'] = None

        with col2:
            st.button(f"Remove Segment {i+1}", on_click=remove_segment, args=(i,), key=f"remove_btn_{i}")
        

st.write("---")
if st.button("Generate Full Sequence Plots", use_container_width=True):
    try:
        fig = arbitrary_sequence_modulation_visualization_streamlit(
            sequence_definition=st.session_state.sequence_definition,
            V_pi=V_pi_eom,
            optical_freq_0=optical_frequency_0_hz,
            num_points_per_cycle=num_points_per_cycle
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.markdown(r"""
This simulator models how a sequence of voltages applied to an Electro-Optic Modulator (EOM) shapes the frequency spectrum of a laser.

* **From Voltage to Phase (Plots 2 & 3):** The modulating voltage ($V$) you define in **Plot 2** is applied to the EOM. This induces a proportional phase shift ($\Delta\phi$) on the optical carrier, as shown in **Plot 3**. The relationship is governed by $\Delta\phi = \pi \cdot (V / V_{\pi})$, where $V_{\pi}$ is the voltage required for a $\pi$ (180°) phase shift. A linear voltage ramp creates a linear phase ramp.

* **From Phase to Frequency (Plot 6):** The resulting frequency spectrum, shown in the FFT in **Plot 6**, is a direct consequence of this phase modulation. A continuous, repeating phase ramp (a serrodyne modulation) shifts the carrier frequency. A constant phase leaves the carrier frequency unchanged.

* **Spectral Characteristics:**
    * **Sideband Spacing:** The complete pattern you define has a total duration ($T_{total}$). This periodicity causes the individual frequency lines in the FFT to be spaced by $1 / T_{total}$.
    * **Linewidth:** The time spent generating a specific frequency component (e.g., the duration of a ramp segment, $T_{ramp}$) determines its spectral width. The Full Width at Half Maximum (FWHM) of this peak is roughly $1 / T_{ramp}$. As an example, if a ramp segment lasts 40 ns, the resulting frequency peak will be broadened by about 1/40 ns = 25 MHz.
    * **Trade-off:** You can create a more sharply defined frequency peak (narrower linewidth) by increasing the duration of its corresponding ramp segment. However, this increases the total pattern repeat time, which in turn makes the frequency sidebands closer together.
""")
