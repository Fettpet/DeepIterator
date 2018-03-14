# DeepIterator
The purpose of this project is to provide an interface for a hierarchical iterator. On of the simplest example for a hierarchical data structure is 

    std::vector< std::vector< int > >

The deepiterator iterates over all int's within the structure.  This simple example has two **layers**. The first layer is a std::vector of std::vector < int >.  The second layer is a std::vector of int`s. To get all the int's the behavior of two layers must be described.  This is done by defining a **Prescription**.  A prescription consists of a navigator, an accessor and a optional prescription. The **accessor** describes the way, a value is get out of the container.  The **navigator** has functionality to go to move through the container. A prescription describes an container free description of the movement of the Iterator through the data. The advantage is, it divide the description of the iteration from the description of the container. Each prescription describes one layer of the hierarchical data structure.
A **view** binds a prescription to a container. A view has functionality similar to the container of the standard template library. This means it has a begin(), rbegin(), end() and rend() function.  Each of this function returns a **DeepIterator**.  For an easier usage of the DeepIterator we had some **make functions** implemented. To use these functions, you need to specify all traits.


## Example
The project delivers some make functions. We use the functions to iterate over each int inside the hierarchical container.

    #include "deepiterator/DeepIterator.hpp"
    // create the prescription to get all int's out of std::vector<int>
    auto && prescriptionSingleLayer = hzdr::makeIteratorPrescription 
        hzdr::makeAccessor(),
        hzdr::makeNavigator( 
            Offset(0),
            Jumpsize(1)
        )
    );
    // create the prescription to get all ints out of std::vector<std::vector<int >>
    auto && prescriptionDoubleLayer = hzdr::makeIteratorPrescription(
        hzdr::makeAccessor(),
        hzdr::makeNavigator(
            Offset(0),
            Jumpsize(1)
        ),
        // the recursive definition
        prescriptionSingleLayer
    );
    std::vector<std::vector<int> > container;
    // binds a container to a prescription
    auto && view = makeView(
        container,
        prescriptionDoubleLayer
    )
      
     // outputs all int's
    for(auto && it=view.begin(); it != view.end(); ++it)
    {
      std::cout << *it;
    }
    // outputs all int's reverse
    for(auto && it=view.rbegin(); it != view.rend(); --it)
    {
      std::cout << *it;
    }
    
More examples are in the Test directory.    

## Installation
The DeepIterator Library is a header only library. You need to add the following lines to your cmake project.

    find_package(DeepIterator)
    include_directories(SYSTEM ${DeepIterator_INCLUDE_DIRS})

