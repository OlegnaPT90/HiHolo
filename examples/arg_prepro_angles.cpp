#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "holo_recons.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("data_prepro_angles");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_file", "-i")
           .help("input hdf5 file of raw detector data")
           .required();

    program.add_argument("--output_file", "-o")
           .help("output hdf5 file of preprocessed data")
           .required();
           
    program.add_argument("--batch_size", "-b")
           .help("batch size of angles processed at a time")
           .required().scan<'i', int>();

    program.add_argument("--kernel_size", "-k")
           .help("kernel size for removing outliers")
           .default_value(3).scan<'i', int>();

    program.add_argument("--threshold", "-t")
           .help("threshold for removing outliers")
           .default_value(2.0f).scan<'g', float>();

    program.add_argument("--range_rows", "-r")
           .help("range of rows to remove stripes")
           .default_value(0).scan<'i', int>();

    program.add_argument("--range_cols", "-c")
           .help("range of columns to remove stripes")
           .default_value(0).scan<'i', int>();

    program.add_argument("--movmean_size", "-m")
           .help("size of moving average for removing stripes")
           .default_value(5).scan<'i', int>();

    program.add_argument("--removal_method", "-M")
           .help("calculation method for removing stripes")
           .default_value("mul");
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::vector<hsize_t> dims;
    std::string input = program.get<std::string>("-i");
    U16Array dark, flat;
    IOUtils::readSingleGram(input, "dark", dark, dims);
    IOUtils::readSingleGram(input, "flat", flat, dims);
    IOUtils::readDataDims(input, "data", dims);
    if (dims.size() != 4) {
        throw std::runtime_error("Invalid holograms or dimensions!");
    }

    int totalAngles = static_cast<int>(dims[0]);
    int numHolograms = static_cast<int>(dims[1]);
    int rows = static_cast<int>(dims[2]);
    int cols = static_cast<int>(dims[3]);
    IntArray imSize {rows, cols};
    
    int batchSize = program.get<int>("-b");
    U16Array rawData(batchSize * numHolograms * rows * cols);

    int kernelSize = program.get<int>("-k");
    float threshold = program.get<float>("-t");
    int rangeRows = program.get<int>("-r");
    int rangeCols = program.get<int>("-c");
    int movmeanSize = program.get<int>("-m");
    std::string method = program.get<std::string>("-M");

    std::string output = program.get<std::string>("-o");
    if (input == output) {
        throw std::runtime_error("Input and output file cannot be the same!");
    }
    IOUtils::createFileDataset(output, "holodata", dims);

    auto preprocessor = PhaseRetrieval::Preprocessor(batchSize, numHolograms, imSize, dark, flat,
                                                     kernelSize, threshold, rangeRows, rangeCols,
                                                     movmeanSize, method);
    FArray holograms;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < totalAngles / batchSize; i++) {
       std::cout << "Processing batch " << i << std::endl;
       IOUtils::read4DimData(input, "data", rawData, i * batchSize, batchSize);
       holograms = preprocessor.processBatch(rawData);
       IOUtils::write4DimData(output, "holodata", holograms, dims, i * batchSize);
    }
        
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}