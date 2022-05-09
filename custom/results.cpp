/**
 *
 * Add function so save result here
 *
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <ctime>
#include <sys/time.h>
#include <sstream>
#include <filesystem>
#include <cstdlib>

using namespace std;

class Save
{
private:
    ofstream prediction_stream;
    ofstream scores_stream;
    array<array<short, 10>, 10> scores;
    string directory_name;
    string timestamp;
    bool empty = true;
    int _neighbors;
    int _size;
    unsigned int misclassified;
    string _weights;

public:
    Save(string dirname, int neighbors, string weights)
    {
        stringstream time_caster;
        time_caster << time(nullptr);
        timestamp = time_caster.str();

        _neighbors = neighbors;
        _weights = weights;

        directory_name = dirname;
        if (directory_name.back() != '/')
        {
            directory_name += '/';
        }

        for (short i = 0; i < 10; i++)
        {
            for (short j = 0; j < 10; j++)
            {
                scores[i][j] = 0;
            }
        }

#ifndef __INTELLISENSE__
        std::filesystem::create_directories((directory_name + timestamp).c_str());
#endif
        prediction_stream.open(directory_name + timestamp + "/knn_prediction-" + timestamp + ".txt");
        scores_stream.open(directory_name + timestamp + "/scores-" + timestamp + ".json");
    }
    void push(short prediction, short expectation)
    {
        if (!empty)
            prediction_stream << ';';
        else
            empty = false;
        prediction_stream << prediction;
        scores[prediction][expectation]++;

        _size++;
        if (prediction != expectation)
            misclassified++;
    }
    void close()
    {
        prediction_stream.close();
        scores_stream << "{";
        scores_stream << "\"total\":" << _size << ',';
        scores_stream << "\"n_neighbors\":" << _neighbors << ',';
        scores_stream << "\"weights\":\"" << _weights << "\",";
        scores_stream << "\"misclassified\":" << misclassified << ',';
        scores_stream << "\"result\":{";
        for (short i = 0; i < 10; i++)
        {
            scores_stream << '"' << i << "\":{";
            for (short j = 0; j < 10; j++)
            {
                scores_stream << '"' << j << '"' << ':' << scores[i][j];
                if (j < 9)
                    scores_stream << ",";
            }
            scores_stream << "}";
            if (i < 9)
                scores_stream << ",";
        }
        scores_stream << "}}";
    }
    int size()
    {
        return _size;
    }
};