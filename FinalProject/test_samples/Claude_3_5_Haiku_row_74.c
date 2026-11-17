#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

#define NUM_CITIES 78
#define MAX_ITERATIONS 1000

// City coordinates structure
typedef struct {
    double x;
    double y;
} City;

// Function prototypes
double calculateDistance(City city1, City city2);
double calculateTotalTourLength(City cities[], int tour[]);
void greedyTSP(City cities[], int tour[]);
void printTour(int tour[]);
void readCityCoordinates(City cities[]);

// Main function
int main() {
    City cities[NUM_CITIES];
    int tour[NUM_CITIES];
    double bestTourLength = DBL_MAX;
    int bestTour[NUM_CITIES];

    // Seed random number generator
    srand(42);

    // Read city coordinates
    readCityCoordinates(cities);

    // Run multiple iterations to find a better solution
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        // Solve TSP using greedy nearest neighbor heuristic
        greedyTSP(cities, tour);

        // Calculate total tour length
        double currentTourLength = calculateTotalTourLength(cities, tour);

        // Update best tour if current tour is shorter
        if (currentTourLength < bestTourLength) {
            bestTourLength = currentTourLength;
            for (int i = 0; i < NUM_CITIES; i++) {
                bestTour[i] = tour[i];
            }
        }
    }

    // Print results
    printf("Best Tour Length: %.2f\n", bestTourLength);
    printf("Best Tour Order:\n");
    printTour(bestTour);

    return 0;
}

// Calculate Euclidean distance between two cities
double calculateDistance(City city1, City city2) {
    double dx = city1.x - city2.x;
    double dy = city1.y - city2.y;
    return sqrt(dx * dx + dy * dy);
}

// Calculate total tour length
double calculateTotalTourLength(City cities[], int tour[]) {
    double totalLength = 0.0;
    for (int i = 0; i < NUM_CITIES - 1; i++) {
        totalLength += calculateDistance(cities[tour[i]], cities[tour[i + 1]]);
    }
    totalLength += calculateDistance(cities[tour[NUM_CITIES - 1]], cities[tour[0]]);
    return totalLength;
}

// Greedy nearest neighbor TSP solver
void greedyTSP(City cities[], int tour[]) {
    bool visited[NUM_CITIES] = {false};
    int startCity = rand() % NUM_CITIES;

    tour[0] = startCity;
    visited[startCity] = true;

    for (int i = 1; i < NUM_CITIES; i++) {
        int currentCity = tour[i - 1];
        double minDistance = DBL_MAX;
        int nearestCity = -1;

        // Find nearest unvisited city
        for (int j = 0; j < NUM_CITIES; j++) {
            if (!visited[j]) {
                double distance = calculateDistance(cities[currentCity], cities[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCity = j;
                }
            }
        }

        tour[i] = nearestCity;
        visited[nearestCity] = true;
    }
}

// Print tour order
void printTour(int tour[]) {
    for (int i = 0; i < NUM_CITIES; i++) {
        printf("%d ", tour[i]);
    }
    printf("\n");
}

// Read city coordinates (dummy implementation)
void readCityCoordinates(City cities[]) {
    // For demonstration, generate random coordinates
    for (int i = 0; i < NUM_CITIES; i++) {
        cities[i].x = (double)rand() / RAND_MAX * 1000.0;
        cities[i].y = (double)rand() / RAND_MAX * 1000.0;
    }
}