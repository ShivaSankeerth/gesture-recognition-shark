from flask import Flask, request
from flask import render_template
import time
import json
import math
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import interp1d
import numpy as np

app = Flask(__name__)

# Calculating constants beforehand to get speed boost
num_sample_points = 100
common_spaced_numbers_100 = np.linspace(0, 1, num_sample_points)

# Calculate alphas for location score
alphas = np.zeros((num_sample_points))
mid_point = num_sample_points // 2
for i in range(mid_point):
    x = i/2450
    alphas[mid_point - i - 1], alphas[mid_point + i] = x, x

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.
    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling (10 points)

    sample_points_X, sample_points_Y = [], []
    
    # Edge Case 1: If gesture/template already has 100 points
    if len(points_X)==100:
        return points_X, points_Y
    
    #Edge Case 2: If gesture/template has only one point the coordinates of the sampled 100 points would be same
    elif len(points_X)==1:
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
        return sample_points_X, sample_points_Y

    # Calculate the euclidean distance between consecutive points
    consecutive_diff  = np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2
    euclidean_distance = np.sqrt(consecutive_diff)

    # Calculate the cumulative distance
    cumulative_distance = np.cumsum(euclidean_distance)

    # Normalize the cumulative distance between 0 and 1
    total_distance = cumulative_distance[-1]
    normalized_cum_dists = cumulative_distance / total_distance

    # Doing Interpolation
    func_x, func_y = interp1d(normalized_cum_dists, points_X, kind='linear'), interp1d(normalized_cum_dists, points_Y, kind='linear')

    # Create the sample points for X and Y
    sample_points_X, sample_points_Y = func_x(common_spaced_numbers_100), func_y(common_spaced_numbers_100)
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

# Normalizing template points
L = 200
# Calculate scaling factor S
templates_width = np.max(template_sample_points_X, axis=1) - np.min(template_sample_points_X, axis=1)
templates_height = np.max(template_sample_points_Y, axis=1) - np.min(template_sample_points_Y, axis=1)
s = L / np.maximum(1, np.max(np.array([templates_width, templates_height]), axis=0))

# Scaling the template sample points
scaling_matrix = np.diag(s)
scaled_template_x,scaled_template_y = np.matmul(scaling_matrix, template_sample_points_X),np.matmul(scaling_matrix, template_sample_points_Y)

# Calculate translation factor tx and ty
template_x_scaled_centroid, template_y_scaled_centroid = np.mean(scaled_template_x, axis=1), np.mean(scaled_template_y, axis=1)
tx, ty = 0 - template_x_scaled_centroid, 0 - template_y_scaled_centroid

# Translate the points
translation_X,translation_Y = np.reshape(tx, (-1, 1)), np.reshape(ty, (-1, 1))
normalized_template_sample_points_X = translation_X + scaled_template_x
normalized_template_sample_points_Y = translation_Y + scaled_template_y


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold

    threshold = 20
    
    # creating a numpy array out of gesture start and end points for faster computation
    gesture_point_start = np.array([gesture_points_X[0], gesture_points_Y[0]])
    gesture_point_end = np.array([gesture_points_X[-1], gesture_points_Y[-1]])

    
    # Gather the start points and end points of templates in a numpy matrix [[x1, y1], [x2, y2], ..., [xn, yn]]
    num_templates = len(template_sample_points_X)
    template_points_start,template_points_end = [], []
    for i in range(num_templates):
        template_points_start.append([template_sample_points_X[i][0], template_sample_points_Y[i][0]])
        template_points_end.append([template_sample_points_X[i][-1], template_sample_points_Y[i][-1]])
    
    template_points_start,template_points_end = np.array(template_points_start),np.array(template_points_end)

    # Calculate distance between gesture and template points for both start and end points
    start_distances = euclidean_distances(np.reshape(gesture_point_start, (1, -1)), template_points_start)[0]
    end_distances = euclidean_distances(np.reshape(gesture_point_end, (1, -1)), template_points_end)[0]

    # Get indices whose start + end distances are less than the threshold
    valid_indices = np.where((start_distances <= threshold) & (end_distances < threshold))[0]

    # Gather valid template sample points, valid words and their respective probabilities using the valid indices
    valid_template_sample_points_X,valid_template_sample_points_Y = np.array(template_sample_points_X)[valid_indices], np.array(template_sample_points_Y)[valid_indices]
    valid_words = [words[valid_index] for valid_index in valid_indices]
    valid_probabilities = [probabilities[valid_word] for valid_word in valid_words]

    # TODO: Do pruning (10 points)

    return valid_words,valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y, valid_indices

def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W)

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)
    return scaled_X, scaled_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y,valid_indices):
    '''Get the shape score for every valid word after pruning.
    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.
    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    
    L = 200

    # Calculate scaling factor 'S'
    width,height = np.max(gesture_sample_points_X) - np.min(gesture_sample_points_X), np.max(gesture_sample_points_Y) - np.min(gesture_sample_points_Y)
    S = L / max(width, height, 1)

    # Scale the points
    scaling_matrix = np.array([[S, 0],[0, S]])
    scaled_gesture_points = np.matmul(scaling_matrix, np.array([gesture_sample_points_X,gesture_sample_points_Y]))

    # Calculate translation factor for X and Y coordinates
    scaled_gesture_centroid_X, scaled_gesture_centroid_Y = np.mean(scaled_gesture_points[0]), np.mean(scaled_gesture_points[1])
    t_y,t_x =  0 - scaled_gesture_centroid_Y,0 - scaled_gesture_centroid_X

    # Translate the points
    normalized_gesture_sample_points = np.array([[t_x],[t_y]]) + scaled_gesture_points

    valid_norm_template_sample_points_X = normalized_template_sample_points_X[valid_indices]
    valid_norm_template_sample_points_Y = normalized_template_sample_points_Y[valid_indices]

    # Calculating (xi - xj) ** 2
    x = (valid_norm_template_sample_points_X - np.reshape(normalized_gesture_sample_points[0], (1, -1))) ** 2
    # Calculating (yi - yj) ** 2
    y = (valid_norm_template_sample_points_Y - np.reshape(normalized_gesture_sample_points[1], (1, -1))) ** 2
    # Calculating square root of (xi - xj)^2 + (yi - yj)^2
    distances = (x + y) ** 0.5

    # Calculate shape scores as mean of distances
    shape_scores = np.sum(distances, axis=1) / num_sample_points

    # TODO: Calculate shape scores (10 points)

    return shape_scores

def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance-r , 0)
        final_max += local_max
    return final_max

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.
    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.
    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :return:
        A list of location scores.
    '''
    location_scores = []
    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return location_scores
    location_scores = np.zeros((len(valid_template_sample_points_X))) # Initialize location scores
    radius = 15
    
    # Creating a list of gesture points [[xi, yi]]
    gesture_points = []
    for i in range(num_sample_points):
        gesture_points.append([gesture_sample_points_X[i], gesture_sample_points_Y[i]])

    for i in range(len(valid_template_sample_points_X)):
        # Create a list of template points
        template_points = []
        for j in range(num_sample_points):
            template_points.append([valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]])

        # Calculate distance of each gesture point with each template point
        euclidean_distance = euclidean_distances(gesture_points, template_points)

        # Find the distance of the closest gesture point to each template point and vice-versa
        template_gesture_closest,gesture_template_closest = np.min(euclidean_distance, axis=0),np.min(euclidean_distance, axis=1)
    
        # If any gesture point is not within the radius tunnel or any template point is not within the radius tunnel
        if np.any(gesture_template_closest - radius > 0) or np.any(template_gesture_closest - radius > 0):
            # Calculate delta as the distance of each gesture point with corresponding template point
            delta = np.diagonal(euclidean_distance)
            # Calculate location score as sum of product of alpha and delta for each point
            location_scores[i] = np.sum(np.multiply(alphas, delta))

    # TODO: Calculate location scores (10 points)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.2
    # TODO: Set your own location weight
    location_coef = 1 - shape_coef
    integration_scores = shape_coef * shape_scores + location_coef * location_scores
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.
    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.
    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 4 #Enter Value Here
    suggestion = ""
    
    sortedIndex = np.argsort(np.array(integration_scores))
    word_int_score_dict = {}
    # print(valid_words)
    for i in range(n):
        try:
            word_int_score_dict[valid_words[sortedIndex[i]]] = integration_scores[sortedIndex[i]]
        except:
            pass
    
    final_score = float('inf')

    for word, int_score in word_int_score_dict.items():
        if (final_score>int_score*(1-probabilities[word])):
            final_score = int_score*(1-probabilities[word])
            best_word = word

    if best_word == " ":
        return "No Match Found"
    
    word_int_score_dict.pop(best_word)

    other_suggestions = list(word_int_score_dict.keys())
    
    suggestion = ' , '.join(other_suggestions)

    # TODO: Get the best word (10 points)
    
    return best_word,suggestion
    


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    # gesture_points_X = [gesture_points_X]
    # gesture_points_Y = [gesture_points_Y]


    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words,valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y,valid_indices = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores( gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y,valid_indices)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word, suggestions = get_best_word(valid_words, integration_scores)


    end_time = time.time()

    return '{"best_word": "' + best_word + '","other_suggestions":"'+suggestions+'", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'

if __name__ == "__main__":
    app.run()