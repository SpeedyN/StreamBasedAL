// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 or the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2016
 * Dep. Of Computer Science
 * Technical University of Munich (TUM)
 *
 */

#ifndef STREAM_BASED_AL_UTILITIES_H_
#define STREAM_BASED_AL_UTILITIES_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <armadillo>  /**< Matrix, vector library */
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <list>

/*
 * Used to generate random numbers
 */
#include <boost/random/exponential_distribution.hpp> 
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>  /**< Random generator mt19937 */
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_real.hpp>

//#define eps 0.00001
#define eps 0.0001

using namespace std;

/* 
 * Convert integer to string
 */
template <typename T>
inline string numberToString(T pNumber)
{
  ostringstream oOStrStream;
  oOStrStream << pNumber;

  return oOStrStream.str();
}

/*
 * Get new file name
 */
inline string new_name(const string str, int numb) {

    int pos = (int) str.find(".",8);
    int tmp_pos = pos;
    string new_str;
    string end = str.substr(pos, str.size());
    /* There are no digits in current file */
    if (!isdigit(str[tmp_pos-1])) {
        cout << "[ERROR]: current filename has no digit number" << endl;
        new_str = str;
        exit(EXIT_FAILURE);
    } else {
        while (isdigit(str[--tmp_pos])) {
            if (tmp_pos <= 0) {
                tmp_pos = pos;
                break;
            }
            //++tmp_pos;
            string file_tmp = str.substr(0, tmp_pos);
            new_str = file_tmp + numberToString(numb) + end;
        }
    }
    return new_str;
}

/*
 * Compares two float values (=)
 * (works only if small values are used)
 */
inline bool equal(float A, float B) {
    if (A == B)  /* Infinite values */
        return true;
    if (fabs(A-B) < eps)
        return true;
    return false;
}
/*
 * Compares two float values (>)
 * (works only if small values are used)
 */
inline bool greater_zero(float A) {
    if (A > eps)
        return true;
    return false;
}

/*
 * Function checks if all elements have the same value
 */
inline bool equal_elements(arma::fvec& prob) {
    bool same = false;
    if (prob.size() > 1) {
        float tmp_point = prob[0];
        unsigned int count = 1;
        for (unsigned int n = 1; n < prob.size(); n++) {
            float new_point = prob[n];
            if (equal(tmp_point, new_point)) {
                count++;
            }
            tmp_point = new_point;
        }
        if (count == prob.size())
            same = true;
    }   
    return same;
}
#endif /* STREAM_BASED_AL__UTILITIES_H_ */
