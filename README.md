# DeepIterator
The purpose of this project is to provide an interface for a hierarchical iterator. On of the simplest example for a hierarchical data structure is std::vector< std::vector< int > >. The deepiterator iterates over all ints within the structure. 
To use the DeepIterator, you must define a view. A view consists of a prescription and a container. A prescription is a recursive describition of the behaiviour of the iterator. Each prescripition describes one layer of the hierachical data strucutre. Two or more prescription can put together. A prescription is independent of a container and consists of a accessor and a navigator. 
The accessor describes the way, a value is get out of the container. 
The navigator has functionality to go to move through the container. 

## Example
The project delivers some make functions. We use the functions to iterate over each int inside the hierachicale container.
auto && prescriptionSingleLayer = hzdr::makeIteratorPrescription(
    hzdr::makeAccessor(),
    hzdr::makeNavigator( 
        Offset(0),
        Jumpsize(1)
    )
);
auto && prescriptionDoubleLayer = hzdr::makeIteratorPrescription(
    hzdr::makeAccessor(),
    hzdr::makeNavigator(
        Offset(0),
        Jumpsize(1)
    ),
    prescriptionSingleLayer
);
std::vector<std::vector<int> > container;
auto && view = makeView(
    container,
    prescriptionDoubleLayer
)
  
for(auto && it=view.begin(); it != view.end(); ++it)
{
  std::cout << *it;
}

## Traits
To use the make functions you need to specify up to 18 traits. Some are grouped in categories

### Categorie
A lot of containers have similar behaiviours. To abuse this knowledge we introduce categories.
