import os
import customtkinter as ctk
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title="", **kwargs):
        super().__init__(master, **kwargs)
        self.title = title
        self.is_expanded = False  # 默认为关闭状态

        # 创建标题按钮
        self.toggle_button = ctk.CTkButton(
            self,
            text=f"▶ {self.title}",  # 默认显示右箭头
            command=self.toggle,
            anchor="w",
            height=30,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30")
        )
        self.toggle_button.pack(fill="x", padx=5, pady=2)

        # 创建内容框架
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        # 初始状态不显示内容框架

    def toggle(self):
        if self.is_expanded:
            self.content_frame.pack_forget()
            self.toggle_button.configure(text=f"▶ {self.title}")
        else:
            self.content_frame.pack(fill="both", expand=True)
            self.toggle_button.configure(text=f"▼ {self.title}")
        self.is_expanded = not self.is_expanded

    def add_item(self, item, command=None):
        btn = ctk.CTkButton(
            self.content_frame,
            text=item,
            anchor="w",
            height=25,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            command=command
        )
        btn._current_fg_color = "transparent"  # 保存原始颜色
        return btn

    def highlight_button(self, button, highlight=True):
        if highlight:
            button.configure(fg_color=("gray70", "gray30"))
        else:
            button.configure(fg_color=button._current_fg_color)

class MainWindow(ctk.CTk):
    # Constants
    WINDOW_TITLE = "Patient Data Viewer"
    PATIENT_LIST_TITLE = "Patient List"
    PLOT_BUTTON_TEXT = "Plot Signals"
    MOVEMENT_PLOT_TITLE_FORMAT_LEFT = "Patient {} Head and Left Eye Movement Signals"
    VELOCITY_PLOT_TITLE_FORMAT_LEFT = "Patient {} Head and Left Eye Velocity Signals"
    MOVEMENT_PLOT_TITLE_FORMAT_RIGHT = "Patient {} Head and Right Eye Movement Signals"
    VELOCITY_PLOT_TITLE_FORMAT_RIGHT = "Patient {} Head and Right Eye Velocity Signals"
    MOVEMENT_PLOT_XLABEL = "Time (s)"
    VELOCITY_PLOT_XLABEL = "Time (s)"
    MOVEMENT_PLOT_YLABEL = "Angle (°)"
    VELOCITY_PLOT_YLABEL = "Velocity (°/s)"
    HEAD_SIGNAL_LABEL = "Head Signal"
    LEFT_EYE_SIGNAL_LABEL = "Left Eye Signal"
    RIGHT_EYE_SIGNAL_LABEL = "Right Eye Signal"

    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.title(self.WINDOW_TITLE)
        self.geometry('800x600')

        # 创建主布局
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 创建侧边栏框架
        self.sidebar_frame = ctk.CTkFrame(self, width=200)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        # 创建可滚动容器
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.sidebar_frame,
            width=180
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 创建可折叠的患者列表
        self.patient_list = CollapsibleFrame(
            self.scrollable_frame,
            title=self.PATIENT_LIST_TITLE
        )
        self.patient_list.pack(fill="x", pady=2)

        # 默认展开患者列表
        self.patient_list.toggle()

        # 在患者列表下方添加滤波器参数控制
        self.filter_frame = CollapsibleFrame(
            self.scrollable_frame,
            title="Filter Parameters"
        )
        self.filter_frame.pack(fill="x", pady=2)

        # 高通滤波参数
        self.highpass_frame = ctk.CTkFrame(self.filter_frame.content_frame)
        self.highpass_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(self.highpass_frame, text="Highpass Filter").pack()

        self.highpass_cutoff = ctk.CTkSlider(
            self.highpass_frame,
            from_=0.01,
            to=1.0,
            number_of_steps=99,
            command=self.update_filter_params
        )
        self.highpass_cutoff.set(0.1)  # 默认值
        self.highpass_cutoff.pack(fill="x", padx=5)

        self.highpass_value_label = ctk.CTkLabel(self.highpass_frame, text="0.1 Hz")
        self.highpass_value_label.pack()

        # 低通滤波参数
        self.lowpass_frame = ctk.CTkFrame(self.filter_frame.content_frame)
        self.lowpass_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(self.lowpass_frame, text="Lowpass Filter").pack()

        self.lowpass_cutoff = ctk.CTkSlider(
            self.lowpass_frame,
            from_=1.0,
            to=20.0,
            number_of_steps=190,
            command=self.update_filter_params
        )
        self.lowpass_cutoff.set(6.0)  # 默认值
        self.lowpass_cutoff.pack(fill="x", padx=5)

        self.lowpass_value_label = ctk.CTkLabel(self.lowpass_frame, text="6.0 Hz")
        self.lowpass_value_label.pack()

        # 添加移动平均窗口大小参数
        self.window_frame = ctk.CTkFrame(self.filter_frame.content_frame)
        self.window_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(self.window_frame, text="Moving Average Window Size").pack()

        self.window_size = ctk.CTkSlider(
            self.window_frame,
            from_=3,
            to=51,
            number_of_steps=24,  # 保证只能选择奇数
            command=self.update_filter_params
        )
        self.window_size.set(5)  # 默认值
        self.window_size.pack(fill="x", padx=5)

        self.window_value_label = ctk.CTkLabel(self.window_frame, text="5 points")
        self.window_value_label.pack()

        # 展开滤波器参数面板
        # self.filter_frame.toggle()

        # 创建主内容区域
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # 在绘图按钮旁边添加关闭图表按钮
        self.button_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.button_frame.pack(fill="x", padx=5, pady=5, side="bottom")

        # 绘图按钮和关闭按钮的框架
        self.plot_control_frame = ctk.CTkFrame(self.button_frame, fg_color="transparent")
        self.plot_control_frame.pack(fill="x", pady=(0, 5))

        # 绘图按钮
        self.plot_button = ctk.CTkButton(
            self.plot_control_frame,
            text=self.PLOT_BUTTON_TEXT,
            command=self.plot_signals,
            height=30
        )
        self.plot_button.pack(fill="x", side="left", expand=True, padx=(0, 2))

        # 关闭图表按钮
        self.close_plots_button = ctk.CTkButton(
            self.plot_control_frame,
            text="Close Plots",
            command=self.close_all_plots,
            height=30,
            fg_color="gray40",
            hover_color="gray30"
        )
        self.close_plots_button.pack(fill="x", side="left", expand=True, padx=(2, 0))

        # 分析信号按钮
        self.analysis_button = ctk.CTkButton(
            self.button_frame,
            text="VOR Analysis",
            command=self.analysis_signals,
            height=30,
            fg_color="#2B7A0B",
            hover_color="#1E5107" 
        )
        self.analysis_button.pack(fill="x")

        # 添加当前选中的患者ID变量
        self.selected_patient = None

        # 添加按钮字典用于管理
        self.patient_buttons = {}
        self.current_highlighted = None

        # 加载患者ID列表
        self.load_patient_ids()

        # 添加右侧图表区域
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.grid_columnconfigure(1, weight=3)  # 让图表区域占据更多空间
        
        # 创建图表容器框架
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.grid_columnconfigure(1, weight=3)
        
        # 创建左右两个图表框架
        self.velocity_frame = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        self.velocity_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.movement_frame = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        self.movement_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 初始时隐藏两个框架
        self.velocity_frame.grid_remove()
        self.movement_frame.grid_remove()
        
        # 创建两个图表
        self.fig_velocity = plt.Figure(figsize=(6, 6))
        self.fig_velocity.patch.set_facecolor('none')  # 设置figure背景透明
        self.canvas_velocity = FigureCanvasTkAgg(self.fig_velocity, master=self.velocity_frame)
        self.canvas_velocity.get_tk_widget().pack(expand=True, padx=10, pady=10)
        self.ax_velocity = self.fig_velocity.add_subplot(111)
        self.ax_velocity.set_facecolor('none')  # 设置axes背景透明
        self.ax_velocity.set_visible(False)
        
        self.fig_movement = plt.Figure(figsize=(6, 6))
        self.fig_movement.patch.set_facecolor('none')  # 设置figure背景透明
        self.canvas_movement = FigureCanvasTkAgg(self.fig_movement, master=self.movement_frame)
        self.canvas_movement.get_tk_widget().pack(expand=True, padx=10, pady=10)
        self.ax_movement = self.fig_movement.add_subplot(111)
        self.ax_movement.set_facecolor('none')  # 设置axes背景透明
        self.ax_movement.set_visible(False)
        
        # 添加滑动条框架到底部
        self.analysis_control_frame = ctk.CTkFrame(self.plot_frame)
        self.analysis_control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        # 添加peak标签
        self.peak_label = ctk.CTkLabel(self.analysis_control_frame, text="Peak Index: -/-")
        self.peak_label.pack(side="left", padx=10)
        
        # 添加peak滑动条
        self.peak_slider = ctk.CTkSlider(
            self.analysis_control_frame,
            from_=0,
            to=1,
            number_of_steps=1,
            command=self.update_analysis_plot
        )
        self.peak_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        # 初始隐藏控制元素
        self.analysis_control_frame.grid_remove()

    def load_patient_ids(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data')

        if os.path.exists(data_path):
            patient_ids = [d for d in os.listdir(data_path)
                         if os.path.isdir(os.path.join(data_path, d))]

            for patient_id in sorted(patient_ids):
                btn = self.patient_list.add_item(
                    patient_id,
                    command=lambda id=patient_id: self.on_patient_select(id)
                )
                btn.pack(fill="x", padx=5, pady=1)
                self.patient_buttons[patient_id] = btn  # 保存按钮引用

    def on_patient_select(self, patient_id):
        # 取消之前的高亮
        if self.current_highlighted:
            self.patient_list.highlight_button(self.patient_buttons[self.current_highlighted], False)

        # 设置新的高亮
        self.patient_list.highlight_button(self.patient_buttons[patient_id], True)
        self.current_highlighted = patient_id
        self.selected_patient = patient_id

    def update_filter_params(self, value=None):
        """更新滤波器参数标签"""
        highpass_value = round(self.highpass_cutoff.get(), 2)
        lowpass_value = round(self.lowpass_cutoff.get(), 1)
        # 确保窗口大小为奇数
        window_value = int(self.window_size.get())
        if window_value % 2 == 0:
            window_value += 1

        self.highpass_value_label.configure(text=f"{highpass_value} Hz")
        self.lowpass_value_label.configure(text=f"{lowpass_value} Hz")
        self.window_value_label.configure(text=f"{window_value} points")

    def plot_signals(self):
        if not self.selected_patient:
            return

        data_path = os.path.join(os.path.dirname(__file__), 'data', self.selected_patient)

        # 读取眼和头部数据
        lefteye_path = os.path.join(data_path, f"{self.selected_patient}_pHIT_lefteye.plist")
        righteye_path = os.path.join(data_path, f"{self.selected_patient}_pHIT_righteye.plist")
        head_path = os.path.join(data_path, f"{self.selected_patient}_pHIT_head.plist")

        # 获取当前滤波器参数
        highpass_params = {
            'cutoff': self.highpass_cutoff.get(),
            'fs': 60,
            'order': 5
        }

        lowpass_params = {
            'cutoff': self.lowpass_cutoff.get(),
            'fs': 60,
            'order': 5
        }

        # 确保窗口大小为奇数
        window_size = int(self.window_size.get())
        if window_size % 2 == 0:
            window_size += 1

        if os.path.exists(lefteye_path) and os.path.exists(head_path):
            # 使用更新后的参数进行信号处理
            lefteye_data, time_lefteye = signal_preprocess(
                lefteye_path,
                highpass_parameter=highpass_params,
                lowpass_parameter=lowpass_params,
                window_size=window_size
            )

            righteye_data, time_righteye = signal_preprocess(
                righteye_path,
                highpass_parameter=highpass_params,
                lowpass_parameter=lowpass_params,
                window_size=window_size
            )

            head_data, time_head = signal_preprocess(
                head_path,
                highpass_parameter=highpass_params,
                lowpass_parameter=lowpass_params,
                window_size=window_size
            )

            lefteye_velocity, time_lefteye_velocity = velocity_calculation(lefteye_data)
            head_velocity, time_head_velocity = velocity_calculation(head_data)
            righteye_velocity, time_righteye_velocity = velocity_calculation(righteye_data)

            # invert head_velocity
            head_velocity = -head_velocity

            # 创建共享x轴的上下两个子图
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # 位置信号图（上图）
            ax1.plot(time_head, head_data, label=self.HEAD_SIGNAL_LABEL)
            ax1.plot(time_lefteye, lefteye_data, label=self.LEFT_EYE_SIGNAL_LABEL)
            ax1.legend()
            ax1.set_title(f'{self.MOVEMENT_PLOT_TITLE_FORMAT_LEFT.format(self.selected_patient)}')
            ax1.set_ylabel(self.MOVEMENT_PLOT_YLABEL)
            ax1.grid(True)

            # 速度信号图（下图）
            ax2.plot(time_head_velocity, head_velocity, label=self.HEAD_SIGNAL_LABEL)
            ax2.plot(time_lefteye_velocity, lefteye_velocity, label=self.LEFT_EYE_SIGNAL_LABEL)
            ax2.legend()
            ax2.set_title(f'{self.VELOCITY_PLOT_TITLE_FORMAT_LEFT.format(self.selected_patient)}')
            ax2.set_xlabel(self.MOVEMENT_PLOT_XLABEL)
            ax2.set_ylabel(self.VELOCITY_PLOT_YLABEL)
            ax2.grid(True)

            # 调子图间距
            plt.tight_layout()

            # Head and right eye velocity plot
            fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax3.plot(time_head, head_data, label=self.HEAD_SIGNAL_LABEL)
            ax3.plot(time_righteye, righteye_data, label=self.RIGHT_EYE_SIGNAL_LABEL)
            ax3.legend()
            ax3.set_title(f'{self.VELOCITY_PLOT_TITLE_FORMAT_RIGHT.format(self.selected_patient)}')
            ax3.set_ylabel(self.VELOCITY_PLOT_YLABEL)
            ax3.grid(True)

            ax4.plot(time_head_velocity, head_velocity, label=self.HEAD_SIGNAL_LABEL)
            ax4.plot(time_righteye_velocity, righteye_velocity, label=self.RIGHT_EYE_SIGNAL_LABEL)
            ax4.legend()
            ax4.set_title(f'{self.VELOCITY_PLOT_TITLE_FORMAT_RIGHT.format(self.selected_patient)}')
            ax4.set_xlabel(self.MOVEMENT_PLOT_XLABEL)
            ax4.set_ylabel(self.VELOCITY_PLOT_YLABEL)
            ax4.grid(True)

            plt.tight_layout()
            plt.show()

    def close_all_plots(self):
        """关闭所有打开的matplotlib图表"""
        plt.close('all')

    def analysis_signals(self):
        if not self.selected_patient:
            return

        data_path = os.path.join(os.path.dirname(__file__), 'data', self.selected_patient)
        lefteye_path = os.path.join(data_path, f"{self.selected_patient}_pHIT_lefteye.plist")
        head_path = os.path.join(data_path, f"{self.selected_patient}_pHIT_head.plist")

        # 获取当前滤波器参数
        highpass_params = {
            'cutoff': self.highpass_cutoff.get(),
            'fs': 60,
            'order': 5
        }

        lowpass_params = {
            'cutoff': self.lowpass_cutoff.get(),
            'fs': 60,
            'order': 5
        }

        window_size = int(self.window_size.get())
        if window_size % 2 == 0:
            window_size += 1

        if os.path.exists(lefteye_path) and os.path.exists(head_path):
            # 处理信号
            lefteye_data, time_lefteye = signal_preprocess(
                lefteye_path,
                highpass_parameter=highpass_params,
                lowpass_parameter=lowpass_params,
                window_size=window_size
            )

            head_data, time_head = signal_preprocess(
                head_path,
                highpass_parameter=highpass_params,
                lowpass_parameter=lowpass_params,
                window_size=window_size
            )

            # 保存位置数据
            self.head_data = head_data
            self.lefteye_data = lefteye_data
            self.time_head = time_head
            self.time_lefteye = time_lefteye

            # 计算速度数据
            self.lefteye_velocity, self.time_lefteye_velocity = velocity_calculation(lefteye_data)
            self.head_velocity, self.time_head_velocity = velocity_calculation(head_data)
            self.head_velocity = -self.head_velocity
            
            self.peaks = velocity_peak_detection(self.head_velocity)
            
            if len(self.peaks) == 0:
                return
                
            # 准备数据字典
            analysis_data = {
                'head_data': self.head_data,
                'lefteye_data': self.lefteye_data,
                'time_head': self.time_head,
                'time_lefteye': self.time_lefteye,
                'head_velocity': self.head_velocity,
                'lefteye_velocity': self.lefteye_velocity,
                'time_head_velocity': self.time_head_velocity,
                'time_lefteye_velocity': self.time_lefteye_velocity,
                'peaks': self.peaks,
                'title_name': 'Left VOR Gain Analysis'
            }
            
            # 创建分析窗口
            analysis_window = AnalysisWindow(self, analysis_data)
            analysis_window.focus()  # 将焦点转移到新窗口
            
            self.head_velocity = -self.head_velocity
            self.lefteye_velocity = -self.lefteye_velocity
            self.peaks = velocity_peak_detection(self.head_velocity)
            
            # 准备数据字典
            analysis_data = {
                'head_data': self.head_data,
                'lefteye_data': self.lefteye_data,
                'time_head': self.time_head,
                'time_lefteye': self.time_lefteye,
                'head_velocity': self.head_velocity,
                'lefteye_velocity': self.lefteye_velocity,
                'time_head_velocity': self.time_head_velocity,
                'time_lefteye_velocity': self.time_lefteye_velocity,
                'peaks': self.peaks,
                'title_name': 'Right VOR Gain Analysis'
            }
            
            # 创建分析窗口
            analysis_window = AnalysisWindow(self, analysis_data)
            analysis_window.focus()  # 将焦点转移到新窗口
            

    def update_analysis_plot(self, index):
        index = int(float(index))
        peak_idx = self.peaks[index]
        window_size = 300
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(self.head_velocity), peak_idx + window_size // 2)
        
        # 更新速度图
        self.ax_velocity.clear()
        self.ax_velocity.set_facecolor('none')  # 保持axes背景透明
        self.ax_velocity.plot(self.time_head_velocity[start_idx:end_idx], 
                             self.head_velocity[start_idx:end_idx], 
                             label='Head Velocity')
        self.ax_velocity.plot(self.time_lefteye_velocity[start_idx:end_idx], 
                             self.lefteye_velocity[start_idx:end_idx], 
                             label='Left Eye Velocity')
        
        self.ax_velocity.set_xlabel('Time (s)')
        self.ax_velocity.set_ylabel('Velocity (°/s)')
        self.ax_velocity.set_title('Velocity Signal')
        self.ax_velocity.grid(True)
        self.ax_velocity.legend()
        
        # 更新位移图
        start_idx_pos = start_idx  # 可能需要调整位移数据的索引
        end_idx_pos = end_idx
        
        self.ax_movement.clear()
        self.ax_movement.set_facecolor('none')  # 保持axes背景透明
        self.ax_movement.plot(self.time_head[start_idx_pos:end_idx_pos], 
                             self.head_data[start_idx_pos:end_idx_pos], 
                             label='Head Position')
        self.ax_movement.plot(self.time_lefteye[start_idx_pos:end_idx_pos], 
                             self.lefteye_data[start_idx_pos:end_idx_pos], 
                             label='Left Eye Position')
        
        self.ax_movement.set_xlabel('Time (s)')
        self.ax_movement.set_ylabel('Position (°)')
        self.ax_movement.set_title('Movement Signal')
        self.ax_movement.grid(True)
        self.ax_movement.legend()
        
        # 调整两个图的布局
        self.fig_velocity.tight_layout()
        self.fig_movement.tight_layout()
        
        self.peak_label.configure(text=f"Peak Index: {index+1}/{len(self.peaks)}")
        
        # 更新两个画布
        self.canvas_velocity.draw()
        self.canvas_movement.draw()

class AnalysisWindow(ctk.CTkToplevel):
    def __init__(self, parent, data):
        super().__init__(parent)
        
        # 设置窗口标题和大小
        self.title(data['title_name'])
        self.geometry("1800x700")
        
        # 保存数据和初始化控制变量
        self.data = data
        self.show_area = False
        
        # 创建主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建三个图表框架在同一行
        self.velocity_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.velocity_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.movement_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.movement_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.peaks_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.peaks_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # 设置列权重使三个图表均匀分布
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)
        
        # 创建三个图表
        self.fig_velocity = plt.Figure(figsize=(6, 6))
        self.fig_velocity.patch.set_facecolor('none')
        self.canvas_velocity = FigureCanvasTkAgg(self.fig_velocity, master=self.velocity_frame)
        self.canvas_velocity.get_tk_widget().pack(expand=True, padx=10, pady=10)
        self.ax_velocity = self.fig_velocity.add_subplot(111)
        self.ax_velocity.set_facecolor('none')
        
        self.fig_movement = plt.Figure(figsize=(6, 6))
        self.fig_movement.patch.set_facecolor('none')
        self.canvas_movement = FigureCanvasTkAgg(self.fig_movement, master=self.movement_frame)
        self.canvas_movement.get_tk_widget().pack(expand=True, padx=10, pady=10)
        self.ax_movement = self.fig_movement.add_subplot(111)
        self.ax_movement.set_facecolor('none')
        
        self.fig_peaks = plt.Figure(figsize=(6, 6))
        self.fig_peaks.patch.set_facecolor('none')
        self.canvas_peaks = FigureCanvasTkAgg(self.fig_peaks, master=self.peaks_frame)
        self.canvas_peaks.get_tk_widget().pack(expand=True, padx=10, pady=10)
        self.ax_peaks = self.fig_peaks.add_subplot(111)
        self.ax_peaks.set_facecolor('none')
        
        # 添加控制框架到底部
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        
        # 添加显示/隐藏面积的按钮
        self.area_button = ctk.CTkButton(
            self.control_frame,
            text="Show VOR Area",
            command=self.toggle_area_display
        )
        self.area_button.pack(side="left", padx=10)
        
        # 添加peak标签
        self.peak_label = ctk.CTkLabel(self.control_frame, text="Peak Index: -/-")
        self.peak_label.pack(side="left", padx=10)
        
        # 添加peak滑动条
        self.peak_slider = ctk.CTkSlider(
            self.control_frame,
            from_=0,
            to=len(data['peaks'])-1,
            number_of_steps=len(data['peaks'])-1,
            command=self.update_plot
        )
        self.peak_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        # 修改平均VOR增益标签为包含标准差的指标标签
        self.metrics_label = ctk.CTkLabel(self.control_frame, text="Metrics: -")
        self.metrics_label.pack(side="right", padx=10)
        
        # 计算所有peaks的VOR增益
        self.gains = []
        for peak_idx in self.data['peaks']:
            window_size = 400
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(self.data['head_velocity']), peak_idx + window_size)
            
            head_velocity_segment = self.data['head_velocity'][start_idx:end_idx]
            lefteye_velocity_segment = self.data['lefteye_velocity'][start_idx:end_idx]
            
            gain, _ = VOR_gain_calculation(head_velocity_segment, lefteye_velocity_segment)
            self.gains.append(gain)
        
        # 计算平均VOR增益和标准差
        self.gains = [gain for gain in self.gains if str(gain) != 'nan' and str(gain) != 'inf']
        self.avg_gain = np.median(self.gains)
        self.gain_std = np.std(self.gains)
        self.metrics_label.configure(text=f"Metrics: {self.avg_gain:.2f} ± {self.gain_std:.2f}")
        
        # 绘制所有peaks对齐图
        self.plot_aligned_peaks()
        
        # 显示第一个peak
        self.peak_slider.set(0)
        self.update_plot(0)
    
    def toggle_area_display(self):
        """切换面积显示状态"""
        self.show_area = not self.show_area
        self.area_button.configure(
            text="Hide VOR Area" if self.show_area else "Show VOR Area"
        )
        self.update_plot(self.peak_slider.get())  # 更新当前图表
    
    def plot_aligned_peaks(self):
        """绘制所有peaks对齐后的叠加图"""
        self.ax_peaks.clear()
        self.ax_peaks.set_facecolor('none')
        
        window_size = 240  # 与单个peak显示窗口大小相同
        half_window = window_size // 2
        
        # 存储所有对齐后的数据
        aligned_head_velocities = []
        aligned_eye_velocities = []
        aligned_times = np.arange(-half_window, half_window) / 60  # 假设采样率为60Hz
        
        # 对齐并收集所有peaks数据
        for peak_idx in self.data['peaks']:
            start_idx = max(0, peak_idx - half_window)
            end_idx = min(len(self.data['head_velocity']), peak_idx + half_window)
            
            # 如果数据长度不足，跳过此peak
            if end_idx - start_idx != window_size:
                continue
                
            aligned_head_velocities.append(self.data['head_velocity'][start_idx:end_idx])
            aligned_eye_velocities.append(self.data['lefteye_velocity'][start_idx:end_idx])
        
        # 转换为numpy数组并计算平均值和标准差
        aligned_head_velocities = np.array(aligned_head_velocities)
        aligned_eye_velocities = np.array(aligned_eye_velocities)
        
        mean_head = np.mean(aligned_head_velocities, axis=0)
        mean_eye = np.mean(aligned_eye_velocities, axis=0)
        std_head = np.std(aligned_head_velocities, axis=0)
        std_eye = np.std(aligned_eye_velocities, axis=0)
        
        # 绘制平均值和标准差区域
        self.ax_peaks.plot(aligned_times, mean_head, 
                          label='Mean Head Velocity', 
                          color='blue', 
                          linewidth=2)
        self.ax_peaks.fill_between(aligned_times, 
                                  mean_head-std_head, 
                                  mean_head+std_head, 
                                  color='blue', 
                                  alpha=0.2)
        
        self.ax_peaks.plot(aligned_times, mean_eye, 
                          label='Mean Eye Velocity', 
                          color='orange', 
                          linewidth=2)
        self.ax_peaks.fill_between(aligned_times, 
                                  mean_eye-std_eye, 
                                  mean_eye+std_eye, 
                                  color='orange', 
                                  alpha=0.2)
        
        # 添加标题和标签
        self.ax_peaks.set_title('Aligned Peaks Average (Shaded Area: ±1 SD)', 
                               fontsize=12, 
                               pad=10)
        self.ax_peaks.set_xlabel('Time (s)', fontsize=10)
        self.ax_peaks.set_ylabel('Velocity (°/s)', fontsize=10)
        
        # 添加网格
        self.ax_peaks.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        self.ax_peaks.legend(loc='upper right', 
                            fontsize=10, 
                            framealpha=0.9)
        
        # 调整布局
        self.fig_peaks.tight_layout()
        self.canvas_peaks.draw()
    
    def update_plot(self, index):
        index = int(float(index))
        peak_idx = self.data['peaks'][index]
        window_size = 240
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(self.data['head_velocity']), peak_idx + window_size // 2)
        
        # 更新速度图
        self.ax_velocity.clear()
        self.ax_velocity.set_facecolor('none')
        
        # 绘制基本的速度曲线
        self.ax_velocity.plot(self.data['time_head_velocity'][start_idx:end_idx], 
                            self.data['head_velocity'][start_idx:end_idx], 
                            label='Head Velocity')
        self.ax_velocity.plot(self.data['time_lefteye_velocity'][start_idx:end_idx], 
                            self.data['lefteye_velocity'][start_idx:end_idx], 
                            label='Left Eye Velocity')
        
        # 计算当前片段的VOR增益和面积信息
        head_velocity_segment = self.data['head_velocity'][start_idx:end_idx]
        lefteye_velocity_segment = self.data['lefteye_velocity'][start_idx:end_idx]
        gain, area_info = VOR_gain_calculation(head_velocity_segment, lefteye_velocity_segment)
        
        # 如果显示面积标志为真，则绘制面积
        if self.show_area:
            # 填充头部速度曲线下的面积
            self.ax_velocity.fill_between(
                self.data['time_head_velocity'][start_idx:end_idx][area_info['head_indices'][0]:area_info['head_indices'][1]], 
                head_velocity_segment[area_info['head_indices'][0]:area_info['head_indices'][1]], 
                alpha=0.3, 
                color='blue', 
                label=f'Head AUC'
            )
            
            # 填充眼睛速度曲线下的面积
            self.ax_velocity.fill_between(
                self.data['time_lefteye_velocity'][start_idx:end_idx][area_info['eye_indices'][0]:area_info['eye_indices'][1]], 
                lefteye_velocity_segment[area_info['eye_indices'][0]:area_info['eye_indices'][1]], 
                alpha=0.3, 
                color='orange', 
                label=f'Eye AUC'
            )
        
        # 设置速度图的标题、标签和网格
        self.ax_velocity.set_title('Velocity Signal', pad=10)
        self.ax_velocity.set_xlabel('Time (s)')
        self.ax_velocity.set_ylabel('Velocity (°/s)')
        self.ax_velocity.grid(True, linestyle='--', alpha=0.7)
        self.ax_velocity.legend(loc='upper right')
        
        # 更新位移图
        self.ax_movement.clear()
        self.ax_movement.set_facecolor('none')
        self.ax_movement.plot(self.data['time_head'][start_idx:end_idx], 
                            self.data['head_data'][start_idx:end_idx], 
                            label='Head Position')
        self.ax_movement.plot(self.data['time_lefteye'][start_idx:end_idx], 
                            self.data['lefteye_data'][start_idx:end_idx], 
                            label='Left Eye Position')
        
        # 设置位移图的标题、标签和网格
        self.ax_movement.set_title('Movement Signal', pad=10)
        self.ax_movement.set_xlabel('Time (s)')
        self.ax_movement.set_ylabel('Position (°)')
        self.ax_movement.grid(True, linestyle='--', alpha=0.7)
        self.ax_movement.legend(loc='upper right')
        
        # 调整图表布局
        self.fig_velocity.tight_layout()
        self.fig_movement.tight_layout()
        
        # 更新标签，显示当前gain和整体统计
        self.peak_label.configure(text=f"Peak Index: {index+1}/{len(self.data['peaks'])}")
        self.metrics_label.configure(text=f"Current Gain: {gain:.2f} | Avg: {self.avg_gain:.2f} ± {self.gain_std:.2f}")
        
        # 更新画布
        self.canvas_velocity.draw()
        self.canvas_movement.draw()

def main():
    app = MainWindow()
    app.mainloop()

if __name__ == '__main__':
    main()