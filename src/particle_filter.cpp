/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	num_particles = 50;

	Particle pAux;

	for (int i = 0; i < num_particles; i++) {

		pAux.id = i;
		pAux.x = dist_x(gen);
		pAux.y = dist_y(gen);
		pAux.theta = dist_theta(gen);
		pAux.weight = 1.0;

		//p.x += N_x_init(gen);
		//p.y += N_y_init(gen);
		//p.theta += N_theta_init(gen);
        weights.push_back(1.0);
		particles.push_back(pAux);
	}

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	default_random_engine gen;


	for (int i = 0; i < num_particles; i++)
	{

		if (fabs(yaw_rate) < 0.00001)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add random noise
		particles[i].x += N_x(gen);
		particles[i].y += N_y(gen);
		particles[i].theta += N_theta(gen);
  	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++)
	{
		double minDist;
		int landmarkId = -1;

		for (unsigned int j = 0; j < predicted.size(); j++) 
		{
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < minDist) 
			{
				minDist = distance;
				landmarkId = predicted[j].id;
			}
		}

		observations[i].id = landmarkId;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html    
    
    for (int n = 0; n < num_particles; n++)
    {
        vector<LandmarkObs> translatedObservations;
        LandmarkObs lMObs;
        for (int i = 0; i < observations.size(); i++)
        {
            LandmarkObs transObs;
            lMObs = observations[i];

            transObs.x = particles[n].x + (lMObs.x * cos(particles[n].theta) - lMObs.y * sin(particles[n].theta));
            transObs.y = particles[n].y + (lMObs.x * sin(particles[n].theta) + lMObs.y * cos(particles[n].theta));
            
            translatedObservations.push_back(transObs);
        }

        particles[n].weight = 1.0;
        
        for(int i = 0; i < translatedObservations.size(); i++)
        {
            double closet_dis = sensor_range;
            int association = 0;
            
            for (int j = 0; j < map_landmarks.landmark_list.size(); j++) 
            {

                double landmark_x = map_landmarks.landmark_list[j].x_f;
                double landmark_y = map_landmarks.landmark_list[j].y_f;

                double calc_dist = sqrt(pow(translatedObservations[i].x-landmark_x,2.0)+pow(translatedObservations[i].y-landmark_y,2.0));
                if(calc_dist < closet_dis)
                {
                    closet_dis = calc_dist;
                    association = j;
                }
            }

            if(association!=0)
            {
                double meas_x = translatedObservations[i].x;
                double meas_y = translatedObservations[i].y;
                double mu_x = map_landmarks.landmark_list[association].x_f;
                double mu_y = map_landmarks.landmark_list[association].y_f;
                long double multipler = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exp(-(pow(meas_x - mu_x,2.0) / (2 * pow(std_landmark[0],2.0)) + pow(meas_y-mu_y,2.0) / (2 * pow(std_landmark[1],2.0))));
                if(multipler > 0)
                {
                    particles[n].weight *= multipler;
                }
            }
        }

        weights[n] = particles[n].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampledParticles;
	default_random_engine gen;

	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);

	double max_weight = *max_element(weights.begin(), weights.end());

	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	for (int i = 0; i < num_particles; i++) {
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampledParticles.push_back(particles[index]);
	}

	particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
