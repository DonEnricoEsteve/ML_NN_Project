from utils import *

def plot_distribution_diff(list_binary_accuracies, list_multi_accuracies,
                           binary_tasks, multiclass_tasks, multiclass_chance):

    # Stack list of accuracies for binary tasks and multi-class tasks
    list_binary_multi = np.vstack((list_binary_accuracies, list_multi_accuracies))

    # Combine binary and multi-class tasks into a single dictionary
    all_conditions = {**binary_tasks, **multiclass_tasks}

    # Prepare to collect results for plotting
    all_differences = []
    all_p_values = []

    # Iterate through the results for each of the 48 combinations of outputs:
    # 6 time windows x 2 sample size x 2 feature size x 2 types of tasks
    for idx, (accuracies, condition_name) in enumerate(zip(list_binary_multi, all_conditions.keys())):
        
        # Determine the chance level for the task
        if condition_name in binary_tasks:
            chance_level = 0.50  # All binary tasks have a chance level of 0.50
        else:
            chance_level = multiclass_chance.get(condition_name)

        # Subtract the chance level from each accuracy
        differences = np.array(accuracies) - chance_level
        all_differences.append(differences)

        # Step 2: Perform the Wilcoxon signed-rank test
        _, p_value = wilcoxon(differences)
        all_p_values.append(p_value)

    # Set up the figure for plotting
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create a horizontal boxplot of the differences for each list
    boxes = ax.boxplot(all_differences, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'),
                       medianprops=dict(color='black', linewidth=2), showmeans=True,
                       meanprops=dict(marker='o', markerfacecolor='red', markersize=8, markeredgecolor='none'))

    # Manually apply custom colors to the specified groups
    for i, box in enumerate(boxes['boxes']):
        if i == 9:
            box.set_facecolor('#f4ccccff') # Food-related
        elif i == 10:
            box.set_facecolor('#fff3c5ff') # Positive-related
        elif i == 11:
            box.set_facecolor('#a9eaf4ff') # Neutral-related
        elif i in range(12,16) or i == 31:
            box.set_facecolor('#ffc1b3ff') # Salient-related
        elif i in range(16,20) or i == 32:
            box.set_facecolor('#a8e6e3ff') # Non-food-related
        elif i in range(20,24) or i == 33:
            box.set_facecolor('#e5bdc2ff') # Control-related
        else:
            box.set_facecolor("#efefefff") # All-related

    # Add asterisks to the appropriate tasks based on p-values
    for idx, p_value in enumerate(all_p_values):
        if p_value < 0.001:
            asterisk = '***'
        elif p_value < 0.01:
            asterisk = '**'
        elif p_value < 0.05:
            asterisk = '*'
        else:
            asterisk = ''

        # Add the asterisk to the right side of the horizontal bars
        if asterisk:
            ax.text(0.255, idx + 1, asterisk, va='center', ha='left', color='black', fontsize=14)

    # Add a chance level line (reference line)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Within-subject Decoding Accuracy = Chance')

    # Create custom legends
    mean_marker = Line2D([0], [0], color='red', marker='o', markersize=6, label='Across-subject decoding accuracy')
    food_related = Line2D([0], [0], color='#f4ccccff', lw=4, label='Single category involved: Food')
    positive_related = Line2D([0], [0], color='#fff3c5ff', lw=4, label='Single category involved: Positive')
    neutral_related = Line2D([0], [0], color='#a9eaf4ff', lw=4, label='Single category involved: Neutral')
    salient_related = Line2D([0], [0], color='#ffc1b3ff', lw=4, label='Salient-related')
    nonfood_related = Line2D([0], [0], color='#a8e6e3ff', lw=4, label='Non-food-related')
    control_related = Line2D([0], [0], color='#e5bdc2ff', lw=4, label='Control-related')
    all_related = Line2D([0], [0], color='#efefefff', lw=4, label='All categories involved')

    # Add titles and labels for both plots
    ax.set_title('Distribution of Differences (Within-Subject Decoding Accuracy - Chance) for All Tasks (n=35) during the m100 Window (10-90%)', 
                 fontdict={'fontsize':12, 'fontweight':'bold'})
    ax.set_xlabel('Difference from Chance Level', fontdict={'fontsize':12, 'fontweight':'bold'})
    ax.set_ylabel('Task', fontdict={'fontsize':12, 'fontweight':'bold'})

    # Add legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [mean_marker, food_related, positive_related, neutral_related, 
                         salient_related, nonfood_related, control_related, all_related],
                         
            labels + ['Across-subject Decoding Accuracy', 'Single category involved: Food', "Single category involved: Positive",
                      'Single category involved: Neutral', "Salient-related", "Non-food-related", 
                      "Control-related", "All categories involved"],

            loc='upper left', bbox_to_anchor=(0, 1))

    # Set x and y limits for better visualization
    ax.set_xlim(-0.2, 0.25)

    # Set custom labels for the boxplot
    y_labels = ["T/S", "T1/S1", "T2/S2",
                "F/N", "F1/N1", "F2/N2",
                "P/C", "P1/C1", "P2/C2",
                "F1/F2", "P1/P2", "T1/T2",
                "F/P", "F1/P1", "F2/P2", "S1/S2",
                "P/T", "P1/T1", "P2/T2", "N1/N2",
                "F/T", "F1/T1", "F2/T2", "C1/C2",
                "1/2", 
                "4-FN-12", "4-TS-12", "4-PC-12",
                "3-FPT", "3-FPT-1", "3-FPT-2",
                "4-FP-12", "4-FT-12", "4-PT-12",
                "6-FPT-12"]
     
    ax.set_yticklabels(y_labels)

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_accuracy_trend(list_binary_acc):
    """
    Plots decoder performance for binary tasks across various time windows as a bar chart.
    Also performs Wilcoxon signed-rank tests to compare means across the 4 labels within each time window.

    Parameters:
    results_dir (str): Directory where the .npy files are stored.
    list_binary_acc (list): List of file names (relative to results_dir), each corresponding to a condition and time window.
    x_labels (list, optional): Custom x-axis labels for the time windows. If None, default labels are used.
    """
    # Define x-axis labels
    if x_labels is None:
        x_labels = [
            'Baseline\n(-300-0)',
            'Post-stimulus\n(0-800)',
            'm100\n(118-155)',
            'm200\n(171-217)',
            'm300\n(239-332)',
            'mLPP\n(350-800)'
        ]

    # Prepare lists to store accuracy data for each condition
    bar_data = []

    # Load and process data from each file in list_binary_acc
    for file in list_binary_acc:
        # Load the .npy file from the results_dir directory
        contrasts = np.load(f'{results_dir}/{file}', allow_pickle=True)
        
        # Extract the 4th inner list (food vs non-food accuracies)
        food_vs_nonfood = contrasts[3]  # This should be the list of accuracies
        
        # Append the accuracies for each time window (assuming the file corresponds to one condition in one time window)
        bar_data.append(food_vs_nonfood)

    # Convert bar_data to a NumPy array for easier manipulation
    bar_data = np.array(bar_data)  # Shape will be (24, 42) after stacking

    # Calculate the mean and standard error of each row (across subjects for each time window-condition pair)
    row_means = np.mean(bar_data, axis=1)  # Shape will be (24,) - mean accuracy for each time window-condition pair
    row_stds = np.std(bar_data, axis=1)    # Shape will be (24,) - standard deviation for each time window-condition pair
    n_subjects = bar_data.shape[1]  # Number of subjects (columns in the bar_data matrix)
    row_sems = row_stds / np.sqrt(n_subjects)  # Standard Error (SE)

    # Now, we need to reshape the row_means and row_sems to group them by the 4 conditions for plotting
    reshaped_means = row_means.reshape((6, 4))  # Reshape to (6, 4), assuming 6 time windows and 4 conditions
    reshaped_sems = row_sems.reshape((6, 4))    # Reshape to (6, 4), assuming 6 time windows and 4 conditions

    # Now, the data should be grouped for each x value with 4 bars in each set.
    width = 0.18  # Bar width
    x_positions = np.arange(len(x_labels))  # x positions for the groups of bars

    # Define the labels and corresponding colors
    labels = ["5-70%", "5-90%", "10-70%", "10-90%"]
    colors = ["#5F8A8B", "#4C9F70", "#F06292", "#E1C34E"]  # Custom colors for each label

    # Create the plot
    _, ax = plt.subplots(figsize=(10, 6))

    # Create the bars for each group with corresponding colors and add error bars
    for i in range(4):  # There are four conditions for each time window
        ax.bar(x_positions + i * (width + 0.025), reshaped_means[:, i], width=width, label=labels[i], color=colors[i])
        ax.errorbar(x_positions + i * (width + 0.025), reshaped_means[:, i], yerr=reshaped_sems[:, i], fmt='none', 
                    color='black', capsize=5, capthick=2, elinewidth=1)  # Error bars (Standard Error)

    # Perform Wilcoxon signed-rank tests and add significance annotations
    for i in range(len(x_labels)):
        # Select the 4 rows corresponding to the 4 conditions in the current time window
        window_data = bar_data[i * 4:(i + 1) * 4, :]  # Shape: (4, 42)
        
        for j in range(4):
            for k in range(j + 1, 4):  # Compare each pair of conditions
                # Perform Wilcoxon signed-rank test between conditions j and k for the raw data of the current time window
                stat, p_value = wilcoxon(window_data[j], window_data[k])  # Paired comparison across subjects
                if p_value < 0.05:
                    # Annotate with a line and significance if p < 0.05
                    ax.plot([x_positions[i] + j * (width + 0.025), x_positions[i] + k * (width + 0.025)],
                            [reshaped_means[i, j], reshaped_means[i, k]], color='black', lw=1)
                    ax.text((x_positions[i] + j * (width + 0.025) + x_positions[i] + k * (width + 0.025)) / 2,
                            max(reshaped_means[i, j], reshaped_means[i, k]) + 0.01,
                            f"p={p_value:.3f}", ha='center', va='bottom', fontsize=9, color='black')

    # Set the labels and title
    ax.set_xlabel('Time Windows (ms)')
    ax.set_ylabel('Across-subject Decoding Accuracy')
    ax.set_title('Decodability of Food vs. Non-food Across Windows and Relative Sample-Feature Size Combinations')

    # Set the x-ticks to be in the center of each group of bars
    ax.set_xticks(x_positions + 1.5 * width) 
    ax.set_xticklabels(x_labels)

    # Add the legend
    ax.legend(title="Pseudo-Trial- \nExplained Var.")

    # Set the y-axis limits
    ax.set_ylim(0.46, 0.58)

    # Add the vertical dashed line between Post-stimulus and m100
    ax.axvline(x=1.8, color='black', linestyle=':', linewidth=1)
    ax.axhline(y=0.5, color='red', linestyle="--", linewidth=1)

    # Enable minor ticks
    ax.minorticks_on()

    # Set minor ticks on the y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  
    ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}')) 

    # Show the plot
    plt.tight_layout()
    plt.show()
    

def calculate_summary_statistics(input_dir, rms_filename, sem_filename, 
                                 time_truncate_ms, sampling_rate_hz=1000):
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
            condition_path = os.path.join(subject_dir, condition + '.npy')  # Assuming data is stored in .npy format
            
            if os.path.exists(condition_path):
                data = np.load(condition_path)  # Shape: (n_trials, n_channels, n_time)
                
                # Average across trials (axis 0)
                data_avg_trials = np.mean(data, axis=0)  # Shape: (n_channels, n_time)
                
                # Compute RMS across channels (axis 0 of the averaged data)
                rms_data = np.sqrt(np.mean(np.square(data_avg_trials), axis=0))  # RMS across channels, keeping time
                print(f"{condition}: {rms_data.shape}")

                # Compute SEM across trials (axis 0)
                sem_data = rms_data / np.sqrt(data.shape[0] * data.shape[1])  # SEM from RMS, keeping time
                
                std_data = np.std(data, axis=0)  # Standard deviation across trials (Shape: n_channels, n_time)
                sem_data = std_data / np.sqrt(data.shape[0])  # SEM across trials, keeping channels and time
                
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


def plot_summary_signals(final_rms, conditions, labels, txt, title):
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

    # Define the new desired range (-300 ms to 800 ms)
    new_start_time = -300  # Start at -300 ms
    new_end_time = 800     # End at 800 ms

    # Plotting the summary statistics
    orig_time_points = np.arange(final_rms[SS_conditions[0]].shape[0])  # Assuming time dimension is the same across conditions

    # Linearly map original time points to the new range
    new_time_points = np.linspace(new_start_time, new_end_time, len(orig_time_points))

    for condition, label in zip(conditions, labels):
        plt.plot(new_time_points, final_rms[condition], label=label)

    # Loop through the highlight regions and add them to the plot
    for start_time, end_time, label in zip(component_start_times, component_end_times, component_labels):
        # Convert start and end times (in seconds) to indices
        highlight_start_idx = np.searchsorted(new_time_points, start_time * 1000)  # Convert start time from seconds to ms
        highlight_end_idx = np.searchsorted(new_time_points, end_time * 1000)  # Convert end time from seconds to ms
        
        # Highlight the region on the plot
        plt.axvspan(new_time_points[highlight_start_idx], new_time_points[highlight_end_idx], color='wheat', alpha=0.3)

        # Calculate the x and y positions for the text inside the highlighted region
        text_x_position = (new_time_points[highlight_start_idx] + new_time_points[highlight_end_idx]) / 2
        text_y_position = plt.gca().get_ylim()[1] * 0.40  # Adjust text position relative to the plot's y-limits
        
        # Add text for the highlighted region
        plt.text(text_x_position, text_y_position, label, color='black', ha='center', va='center', fontsize=9)

    # Add the main text annotation at the top of the plot
    plt.text(-350, plt.gca().get_ylim()[1]*1.01, txt, color='black', ha='center', va='center', fontsize=14, fontweight='bold')

    # Set plot labels and title
    plt.xlabel('Time relative to stimulus onset (ms)')
    plt.ylabel('RMS Amplitude (T)')
    plt.title(f'Descriptive Summary of Evoked Responses by {title} Category across Trials and Channels')

    # Show the legend
    plt.legend(loc='upper right')

    # Set the x-axis limits from -300 ms to 800 ms
    plt.xlim(-300, 800)

    # Show the plot
    plt.show()


def plot_summary_boxplots(final_rms, txt, title, color):

    # # Convert the time range from ms to index positions in the time_points array
    # start_idx = np.searchsorted(np.arange(0,800), start_time_ms)
    # end_idx = np.searchsorted(np.arange(0,800), end_time_ms)

    # Create boxplots for RMS values across conditions (no need to average across time points)
    plt.figure(figsize=(10, 6))

    # Each condition has an array of RMS values for each time point (shape (800,))
    data_to_plot = [final_rms[condition] for condition in SS_conditions]

    # Plot RMS boxplots using seaborn, without fill (no color)
    sns.boxplot(data=data_to_plot, palette="Set2", 
            boxprops=dict(facecolor=color, edgecolor='black'))  # No fill, black edge color

    # Add title, labels, and configure plot
    plt.text(-1.1, plt.gca().get_ylim()[1]*1.02, txt, color='black', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.title(f'Evoked Responses from Original Conditions across Trials, Channels, and Time during the {title}')
    plt.xlabel('Condition')
    plt.xticks(ticks=range(len(SS_conditions)), labels=cond_short_labels, rotation=0)
    plt.ylabel('RMS Amplitude (T)')

    plt.tight_layout()  # Ensure labels fit
    plt.grid(False)
    plt.show()