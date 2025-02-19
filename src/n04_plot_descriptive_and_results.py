from utils import *

def plot_distribution_diff(list_binary_accuracies, list_multi_accuracies,
                  binary_tasks, multiclass_tasks, 
                  binary_chance, multiclass_chance):

    # Stack list of accuracies for binary tasks and multi-class tasks
    list_binary_multi = np.vstack((list_binary_accuracies, list_multi_accuracies))

    # Combine binary and multi-class tasks into a single dictionary
    all_conditions = {**binary_tasks, **multiclass_tasks}

    # Combine binary and multi-class chance levels into a single dictionary
    all_chance_levels = {**binary_chance, **multiclass_chance}

    # Prepare to collect results for plotting
    all_differences = []
    all_p_values = []

    for idx, (accuracies, condition_name) in enumerate(zip(list_binary_multi, list(all_conditions.keys()))):
        # Step 1: Get the chance level for the current condition
        chance_level = all_chance_levels.get(condition_name, 0.50)  # Default to 0.50 if condition name not found

        # Subtract the chance level from each accuracy
        differences = np.array(accuracies) - chance_level
        all_differences.append(differences)

        # Step 2: Perform the Wilcoxon signed-rank test
        stat, p_value = wilcoxon(differences)
        all_p_values.append(p_value)

    # Set up the figure for plotting
    fig, ax = plt.subplots(figsize=(14, 10))

    # Step 3a: Create a Boxplot of the differences for each list
    boxes = ax.boxplot(all_differences, vert=False, patch_artist=True, 
                    boxprops=dict(facecolor='skyblue', color='black'),
                    medianprops=dict(color='black', linewidth=2), showmeans=True,
                    meanprops=dict(marker='o', markerfacecolor='red', markersize=8, markeredgecolor='none'))

    # Apply custom colors to the specified groups
    for i, box in enumerate(boxes['boxes']):
        if i == 0 or i == 4 or i == 5:
            box.set_facecolor('lightgreen')
        elif i == 11 or i == 12 or i == 13 or i == 14:
            box.set_facecolor('lightsalmon')
        elif i == 15 or i == 16:
            box.set_facecolor('#D1A6D0')
        elif i == 17:
            box.set_facecolor('lightgray')

    # Manual asterisk placement (Define which datasets should have which symbols)
    asterisks = {
        0: '*',  
        1: '*',   
        2: '*',  
        3: '*',   
        4: '*',
        5: '*',
        6: '*',
        10: '*',
        11: "*",
        12: "*",
        13: "*"
    }

    # Add the asterisks manually to the bars
    for idx, symbol in asterisks.items():
        ax.text(0.46, idx + 1, symbol, va='center', ha='left', color='black', fontsize=14)

    # Add only one chance level line (reference line)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Within-subject Decoding Accuracy = Chance')

    # Create a custom legend for the mean marker
    mean_marker = Line2D([0], [0], color='red', marker='o', markersize=8, label='Across-subject decoding accuracy')

    # Create a custom legend for the blue boxplots
    blue_box_legend = Line2D([0], [0], color='skyblue', lw=4, label='Binary Task with normal chance levels (0.50)')

    # Create a custom legend for the green boxplots (highlighted tasks)
    green_box_legend = Line2D([0], [0], color='lightgreen', lw=4, label='Binary Task with adjusted chance levels (0.66)')

    # Create a custom legend for the green boxplots (highlighted tasks)
    salmon_box_legend = Line2D([0], [0], color='salmon', lw=4, label='Multi-class Tasks with Chance: 0.333')

    # Create a custom legend for the green boxplots (highlighted tasks)
    lavender_box_legend = Line2D([0], [0], color='#D1A6D0', lw=4, label='Multi-class Tasks with Chance: 0.250')

    # Create a custom legend for the green boxplots (highlighted tasks)
    lightgray_box_legend = Line2D([0], [0], color='lightgray', lw=4, label='Multi-class Tasks with Chance: 0.166')

    # Add titles and labels for both plots
    ax.set_title('Distribution of Differences (Within-Subject Decoding Accuracy - Chance) for All Tasks (200 features) during the Post-Stimulus Window', 
                 fontdict={'fontsize':12, 'fontweight':'bold'})
    ax.set_xlabel('Difference from Chance Level', fontdict={'fontsize':12, 'fontweight':'bold'})
    ax.set_ylabel('Contrast', fontdict={'fontsize':12, 'fontweight':'bold'})

    # Add legend for the chance level line
    # This will ensure that the legend only shows once for the chance level line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [mean_marker, blue_box_legend, green_box_legend, salmon_box_legend, lavender_box_legend, lightgray_box_legend], 
            labels + ['Across-subject Decoding Accuracy', 'Binary Task with Normal Chance: 0.500', 
                        'Binary Task with Adjusted Chance: 0.666', "Multi-class Task with Chance: 0.333",
                        "Multi-class Task with Chance: 0.250", "Multi-class Task with Chance: 0.166"], 
            loc='upper right', bbox_to_anchor=(1, 1))

    # Set x and y limits for better visualization
    ax.set_xlim(-0.35, 0.45)

    # Set custom labels for the boxplot
    y_labels = ["F/N", "F/P", "F/T", "P/T", "F1/N1", "F2/N2", "F1/F2", "P1/P2", "T1/T2", "N1/N2", "1/2", "F/P/T", 
                "F1/P1/T1", "F2/P2/T2", "F1/F2/N1/N2", "F1/F2/P1/P2", "F1/F2/T1/T2", "F1/F2/P1/P2/T1/T2"] 
    ax.set_yticklabels(y_labels)

    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_accuracy_trend(list_binary_acc, list_mult_acc=None, x_labels=None):
    """
    Plots decoder performance for binary tasks across various time windows.
    
    Parameters:
    list_binary_acc (list): List of binary accuracy data for different time windows.
    list_mult_acc (list): List of multi-class accuracy data for different time windows (not used here but included for possible future extension).
    x_labels (list, optional): Custom x-axis labels for the time windows. If None, default labels are used.
    
    """
    # Define default x-axis labels if not provided
    if x_labels is None:
        x_labels = [
            'Post-Stimulus\n(0-800)',
            'm100\n(118-155)',
            'm200\n(171-217)',
            'm300\n(239-332)',
            'mLPP\n(350-800)'
        ]
    
    # Prepare lists to store mean accuracies for each contrast
    mean_food_vs_nonfood = []
    mean_food_1_vs_nonfood_1 = []
    mean_food_2_vs_nonfood_2 = []

    # Loop through the binary accuracy data and calculate the mean for each contrast
    for contrasts in list_binary_acc:
        food_vs_nonfood = contrasts[0]
        food_1_vs_nonfood_1 = contrasts[4]
        food_2_vs_nonfood_2 = contrasts[5]

        mean_food_vs_nonfood.append(np.mean(food_vs_nonfood))
        mean_food_1_vs_nonfood_1.append(np.mean(food_1_vs_nonfood_1))
        mean_food_2_vs_nonfood_2.append(np.mean(food_2_vs_nonfood_2))

    # Plot the means
    plt.plot(x_labels, mean_food_vs_nonfood, label='Food vs Non-food', marker='o', markersize=8, linestyle='-', color='#1f77b4', markeredgewidth=2)
    plt.plot(x_labels, mean_food_1_vs_nonfood_1, label='Food 1 vs. Non-food 1', marker='*', markersize=8, linestyle='--', color='#ff7f0e', markeredgewidth=2)
    plt.plot(x_labels, mean_food_2_vs_nonfood_2, label='Food 2 vs. Non-food 2', marker='p', markersize=8, linestyle=':', color='#2ca02c', markeredgewidth=2)

    plt.xlabel('Time Windows (ms)')
    plt.ylabel('Across-subject Decoding Accuracy')
    plt.title('Decoder Performance for Consistently Decodable Binary Tasks')
    plt.legend()
    plt.tight_layout()  # Adjust layout to fit the labels
    plt.show()

def calculate_summary_statistics(input_dir, rms_filename, sem_filename, 
                                 time_truncate_ms=800, sampling_rate_hz=1000):
    """
    Calculates summary statistics (RMS and SEM) for each condition across all subjects.
    
    Parameters:
    - input_dir: Directory containing all subject directories.
    - conditions: List of conditions to process.
    - time_truncate_ms: Time (in ms) to truncate the data.
    - sampling_rate_hz: Sampling rate (in Hz) of the data.
    
    Returns:
    - final_rms: Dictionary of final RMS values for each condition.
    - final_sem: Dictionary of final SEM values for each condition.
    """
    
    # List all subjects in the directory (assuming they are named with 'sub_' prefix)
    subjects = [d for d in os.listdir(input_dir) if d.startswith('sub_')]
    
    # Initialize a dictionary to hold the summary statistics for each condition
    summary_stats = {condition: {'rms': [], 'sem': []} for condition in SS_conditions}
    
    # Convert time to points (samples)
    time_truncate_points = int(time_truncate_ms * sampling_rate_hz / 1000)
    
    # Loop through each subject and condition to compute summary statistics
    for subject in subjects:
        subject_dir = os.path.join(input_dir, subject)
        
        for condition in SS_conditions:
            condition_path = os.path.join(subject_dir, condition + 'npy')  # Assuming data is stored in .npy format
            
            if os.path.exists(condition_path):
                data = np.load(condition_path)  # Shape: (n_trials, n_channels, n_time)
                
                # Average across trials (axis 0)
                data_avg_trials = np.mean(data, axis=0)  # Shape: (n_channels, n_time)
                
                # Compute RMS across channels (axis 0 of the averaged data)
                rms_data = np.sqrt(np.mean(np.square(data_avg_trials), axis=0))  # RMS across channels, keeping time
                
                # Compute SEM across trials (axis 0)
                sem_data = rms_data / np.sqrt(data.shape[0] * data.shape[1])  # SEM from RMS, keeping time
                
                # Truncate the data up to the desired time point
                rms_data_truncated = rms_data[:time_truncate_points]
                sem_data_truncated = sem_data[:time_truncate_points]
                
                # Append results to the summary_stats dictionary
                summary_stats[condition]['rms'].append(rms_data_truncated)
                summary_stats[condition]['sem'].append(sem_data_truncated)
    
    # Calculate the overall RMS and SEM across all subjects for each condition
    final_rms = {}
    final_sem = {}
    
    for condition in SS_conditions:
        final_rms[condition] = np.mean(summary_stats[condition]['rms'], axis=0)
        final_sem[condition] = np.mean(summary_stats[condition]['sem'], axis=0)
    
    np.save(f'{SS_dir}/{rms_filename}', final_rms, allow_pickle=True)
    np.save(f'{SS_dir}/{sem_filename}', final_sem, allow_pickle=True)
    
    return final_rms, final_sem

def plot_summary_signals(final_rms, final_sem, labels, txt, title):
    """
    Plots the descriptive summary of evoked responses (RMS and SEM) across trials and channels.

    Parameters:
    - final_rms (dict): A dictionary containing RMS values for each condition. 
                         Each key is a condition, and each value is an array of RMS values across time.
    - final_sem (dict): A dictionary containing SEM values for each condition.
                         Each key is a condition, and each value is an array of SEM values across time.
    - labels (list): A list of labels to be used in the plot legend. Each label corresponds to a condition in `final_rms` and `final_sem`.
    - txt (str): Text to be displayed in the plot (e.g., for further clarification or annotations).
    - title (str): Title that will appear in the plot. This will be included in the plot's main title as part of the summary.
    """
    
    plt.figure(figsize=(10, 6))

    # Plotting the summary statistics
    time_points = np.arange(final_rms[SS_conditions[0]].shape[0])  # Assuming time dimension is the same across conditions

    for condition, label in zip(SS_conditions, labels):
        plt.plot(time_points, final_rms[condition], label=label)
        plt.fill_between(time_points, final_rms[condition] - final_sem[condition], 
                         final_rms[condition] + final_sem[condition], alpha=0.3)

    # Loop through the highlight regions and add them to the plot
    for start_time, end_time, label in zip(component_start_times, component_end_times, component_labels):
        # Convert start and end times (in seconds) to indices
        highlight_start_idx = np.searchsorted(time_points, start_time * 1000)  # Convert to ms
        highlight_end_idx = np.searchsorted(time_points, end_time * 1000)  # Convert to ms
        
        # Highlight the region
        plt.axvspan(time_points[highlight_start_idx], time_points[highlight_end_idx], color='wheat', alpha=0.3)

        # Calculate the x and y positions for the text inside the highlighted region
        text_x_position = (time_points[highlight_start_idx] + time_points[highlight_end_idx]) / 2
        text_y_position = plt.gca().get_ylim()[1] * 0.23  # Adjust text position relative to the plot's y-limits
        
        # Add text for the highlighted region
        plt.text(text_x_position, text_y_position, label, color='black', ha='center', va='center', fontsize=12)

    plt.text(-40, plt.gca().get_ylim()[1]*1.01, txt, color='black', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.xlabel('Time relative to stimulus onset (ms)')
    plt.ylabel('RMS Amplitude (T)')
    plt.title(f'Descriptive Summary of Evoked Responses by {title} across Trials and Channels')
    plt.legend(loc='upper right')
    plt.xlim(0, 800)  # Truncate the x-axis to 800 ms
    plt.show()

def plot_summary_boxplots(final_rms, start_time_ms, end_time_ms, txt, title, color):

    # Convert the time range from ms to index positions in the time_points array
    start_idx = np.searchsorted(np.arange(0,800), start_time_ms)
    end_idx = np.searchsorted(np.arange(0,800), end_time_ms)

    # Create boxplots for RMS values across conditions (no need to average across time points)
    plt.figure(figsize=(10, 6))

    # Each condition has an array of RMS values for each time point (shape (800,))
    data_to_plot = [final_rms[condition][start_idx:end_idx] for condition in SS_conditions]

    # Plot RMS boxplots using seaborn, without fill (no color)
    sns.boxplot(data=data_to_plot, palette="Set2", 
            boxprops=dict(facecolor=color, edgecolor='black'))  # No fill, black edge color

    # Add title, labels, and configure plot
    plt.text(-1.1, plt.gca().get_ylim()[1]*1.02, txt, color='black', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.title(f'Descriptive Summary of Evoked Responses across Trials, Channels, and Time during the {title}')
    plt.xlabel('Condition')
    plt.xticks(ticks=range(len(SS_conditions)), labels=SS_labels, rotation=45)
    plt.ylabel('RMS Amplitude (T)')

    plt.tight_layout()  # Ensure labels fit
    plt.grid(False)
    plt.show()