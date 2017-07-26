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
using std::normal_distribution;
using std::default_random_engine;

static default_random_engine ran_gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  normal_distribution<double> norm_dist_x(x, std[0]);
  normal_distribution<double> norm_dist_y(y, std[1]);
  normal_distribution<double> norm_dist_theta(theta, std[2]);

  num_particles = 100;

  particles.resize(num_particles);
  weights.resize(num_particles);

  for( int parti = 0; parti < num_particles; parti++ ){
    particles[parti].id = parti;
    particles[parti].x = norm_dist_x(ran_gen);
    particles[parti].y = norm_dist_y(ran_gen);
    particles[parti].theta = norm_dist_theta(ran_gen);
    particles[parti].weight = 1;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  normal_distribution<double> norm_dist_x(0, std_pos[0]);
	normal_distribution<double> norm_dist_y(0, std_pos[1]);
	normal_distribution<double> norm_dist_theta(0, std_pos[2]);

  for( int parti = 0; parti< num_particles; parti++ ){

    if( abs(yaw_rate) > 0.0000001 ){
      particles[parti].x += (velocity / yaw_rate) * ( sin(particles[parti].theta + yaw_rate*delta_t) - sin( particles[parti].theta ) );
      particles[parti].y += (velocity/yaw_rate) * ( cos(particles[parti].theta) - cos(particles[parti].theta + yaw_rate*delta_t) );
      particles[parti].theta += yaw_rate*delta_t;
    } else{
      particles[parti].x += velocity * delta_t * cos(particles[parti].theta);
      particles[parti].y += velocity * delta_t * sin(particles[parti].theta);
    }


    //Add noise
    particles[parti].x += norm_dist_x(ran_gen);
    particles[parti].y += norm_dist_y(ran_gen);
    particles[parti].theta += norm_dist_theta(ran_gen);

  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  double multi_gauss;

  for( int parti = 0; parti < num_particles; parti++ ){

    multi_gauss = 1;

    for( int obs = 0; obs < observations.size(); obs++ ){

      // Transform observation from vehicle coordintes to map coordinates
      double obs_map_x =  observations[obs].x * cos(particles[parti].theta) - observations[obs].y * sin( particles[parti].theta ) + particles[parti].x;
      double obs_map_y =  observations[obs].x * sin(particles[parti].theta) + observations[obs].y * cos( particles[parti].theta ) + particles[parti].y;

      // Check if this observation is out of sensor range
      if( sqrt(pow(obs_map_x - particles[parti].x, 2) + pow(obs_map_y - particles[parti].y, 2)) > sensor_range ){
        continue;
      }

      vector<double> dist_lm_obs(map_landmarks.landmark_list.size());
      for( int lm = 0; lm < map_landmarks.landmark_list.size(); lm++ ){
        dist_lm_obs[lm] = sqrt(pow(obs_map_x - map_landmarks.landmark_list[lm].x_f, 2) + pow(obs_map_y - map_landmarks.landmark_list[lm].y_f, 2));
      }

      vector<double>::iterator min_ele = min_element(begin(dist_lm_obs), end(dist_lm_obs));
      int pos_min_ele = distance(begin(dist_lm_obs), min_ele);


      multi_gauss *= exp(-0.5 * (pow((map_landmarks.landmark_list[pos_min_ele].x_f - obs_map_x) / std_landmark[0],2) + pow((map_landmarks.landmark_list[pos_min_ele].y_f - obs_map_y) / std_landmark[1],2))) / (2*M_PI*std_landmark[0]*std_landmark[1]);

    }

    particles[parti].weight = multi_gauss;
    weights[parti] = particles[parti].weight;


  }

}

void ParticleFilter::resample() {

  discrete_distribution<int> distri(weights.begin(), weights.end());
  vector<Particle> new_particles (num_particles);

  for( int i = 0; i < num_particles; i++ ){

    new_particles[i] = particles[distri(ran_gen)];

  }

  particles = new_particles;

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
