import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score

def evaluation(trajectory_data):

    def calculate_kinematics(mouse_data):
        distances = []
        displacements = []
        speeds = []
        velocities = []
        accelerations_distance = []
        accelerations_displacement = []
        
        for i in range(1, len(mouse_data)-1):
            x1, y1, t1 = mouse_data[i - 1]
            x2, y2, t2 = mouse_data[i]
            x3, y3, t3 = mouse_data[i + 1]
            
            # Calculate distance between points
            displacement = [x2-x1, y2-y1, t2-t1]
            if t2-t1 == 0:
                displacement[-1] = 1e-4
            distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
            distances.append(distance)
            displacements.append(displacement)
            if i == len(mouse_data)-2:
                displacement_final = [x3-x2, y3-y2, t3-t2]
                if t3-t2 == 0:
                    displacement_final[-1] = 1e-4
                distance_final = np.sqrt(displacement_final[0]**2 + displacement_final[1]**2)
                distances.append(distance)
                displacements.append(displacement)

            # Calculate speed (distance/time)
            speed = distance / displacement[-1]
            velocity = [x / displacement[-1] for x in displacement[0:2]]

            speeds.append(speed)
            velocities.append(velocity)
            if i == len(mouse_data)-2:
                speed_final = distance_final / displacement_final[-1]
                velocity_final = [x / displacement_final[-1] for x in displacement_final[0:2]]
                speeds.append(speed_final)
                velocities.append(velocity_final)
        
        for i in range(1, len(mouse_data)-2):
            # Calculate acceleration (change in speed/time)
            #print(i+1, len(displacements), len(distances), len(speeds))
            #print(type(distances[i]), type(speeds[i]), type(speeds[i]), displacements[i][-1])
            acceleration_distance = 2*(distances[i] - (speeds[i+1] - speeds[i])*displacements[i+1][-1])/(displacements[i][-1])**2
            acceleration_displacement = list(2*(np.array(displacements[i][0:2]) - (np.array(velocities[i+1]) - np.array(velocities[i]))*displacements[i+1][-1])/(displacements[i][-1])**2)
            accelerations_distance.append(acceleration_distance)
            accelerations_displacement.append(acceleration_displacement)
        
        return distances, displacements, speeds, velocities, accelerations_distance, accelerations_displacement

    def calculate_smoothness(mouse_data):
        angles = []
        
        for i in range(2, len(mouse_data)):
            x1, y1, _ = mouse_data[i - 2]
            x2, y2, _ = mouse_data[i - 1]
            x3, y3, _ = mouse_data[i]
            
            # Vectors
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x3 - x2, y3 - y2])
            
            # Calculate the angle between the two vectors
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
            
            angles.append(np.degrees(angle))
        
        # Average angle deviation (larger angles = more jerky movements)
        avg_angle = np.mean(angles) if angles else 0
        return avg_angle

    def calculate_timing_consistency(mouse_data):
        time_differences = []
        
        for i in range(1, len(mouse_data)):
            t1 = mouse_data[i - 1][2]
            t2 = mouse_data[i][2]
            time_diff = t2 - t1
            time_differences.append(time_diff)
        
        # Standard deviation of the time differences
        timing_variability = np.std(time_differences)
        return timing_variability

    def calculate_smoothness_edges(mouse_data):
        angles = []
        
        for i in range(2, len(mouse_data)):
            x1, y1, _ = mouse_data[i - 2]
            x2, y2, _ = mouse_data[i - 1]
            x3, y3, _ = mouse_data[i]
            
            # Vectors
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x3 - x2, y3 - y2])
            
            # Calculate the angle between the two vectors
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
            
            angles.append(np.degrees(angle))
        
        # Average angle deviation (larger angles = more jerky movements)
        avg_angle_start = np.mean(angles[0:5]) if angles else 0
        avg_angle_end = np.mean(angles[-6:]) if angles else 0
        return avg_angle_start, avg_angle_end

    def calculate_position_linear_fit(mouse_data):
        #xi, yi, _ = mouse_data[0]
        #xf, yf, _ = mouse_data[-1]
        #m = (yf-yi)/(xf-xi)
        #x_grid = np.linspace(xi, xf, len(mouse_data))
        #y_exp = m*x_grid + yi*np.ones(len(mouse_data))
        model = LinearRegression()
        model.fit(mouse_data[:,0].reshape(-1, 1), mouse_data[:,1].reshape(-1, 1))
        r2 = model.score(mouse_data[:,0].reshape(-1, 1), mouse_data[:,1].reshape(-1, 1))
        return r2

    def analyze_mouse_data(mouse_data):
        distances, displacements, speeds, velocities, accelerations_distance, accelerations_displacement = calculate_kinematics(mouse_data)
        smoothness = calculate_smoothness(mouse_data)
        timing_consistency = calculate_timing_consistency(mouse_data)
        angle_start, angle_end = calculate_smoothness_edges(mouse_data)
        linear_fit = calculate_position_linear_fit(mouse_data)
        
        # Feature vector
        features = [np.mean(np.array(distances[0:5])-np.array(distances[-5:])), np.mean(np.array(speeds[0:5])-np.array(speeds[-5:])), np.mean(np.array(accelerations_distance[0:5])-np.array(accelerations_distance[-5:])), timing_consistency, smoothness, np.abs(angle_start - angle_end), linear_fit]
        
        #Truth matrix
        truth_matrix = [False,False,False,False,False,False,False]

        # Classify using the trained model
        if np.abs(features[0] - np.mean(distances[int(len(distances)/2)-2:int(len(distances)/2)+3])) > np.std(distances)/np.sqrt(len(distances)):
            truth_matrix[0] = True
        if np.abs(features[1] - np.mean(speeds[int(len(speeds)/2)-2:int(len(speeds)/2)+3])) > np.std(speeds)/np.sqrt(len(speeds)):
            truth_matrix[1] = True
        if np.abs(features[2] - np.mean(accelerations_distance[int(len(accelerations_distance)/2)-2:int(len(accelerations_distance)/2)+3])) > np.std(accelerations_distance)/np.sqrt(len(accelerations_distance)):
            truth_matrix[2] = True
        if np.abs(timing_consistency) > 0.1:
            truth_matrix[3] = True
        if np.abs(smoothness) > 5:
            truth_matrix[4] = True
        if np.abs(angle_start - angle_end) > 2:
            truth_matrix[5] = True
        if linear_fit < .95:
            truth_matrix[6] = True

        if np.sum(truth_matrix) > 3:
            #print("Movement is human-generated.")
            return([True,truth_matrix])
        else:
            #print("Movement is machine-generated.")
            return([False,truth_matrix])

    predictions = []
    visualize = np.zeros(7)
    for i in trajectory_data:
        if analyze_mouse_data(i)[0] == True:
            predictions.append(True)
            #for j in analyze_mouse_data(i)[1]:
            #    if j == True:
            #        visualize[j] += 1
        else:
            predictions.append(False)
    
    #plt.figure("Total Distribution of Truth Values")
    #plt.bar(["Distance","Speed","Acceleration","Timing","Smoothness","Change in Angle", "Linear Fit"], visualize)
    #plt.ylabel('Count')
    #plt.show()
    
    accuracy = (sum(predictions)/len(trajectory_data))

    return (f"Model Accuracy: {accuracy}")