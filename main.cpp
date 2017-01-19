#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <memory>
#include "deepForeach.hpp"
#include "isContainer.hpp"
#include <cstdlib>
#include "Supercell.hpp"
#include "Frame.hpp"
#include "Particle.hpp"
template<
    typename TElement>
struct DeepIterator;


/**
 * @brief Der DeepContainer ist ein virtueller Container, welcher zusätzliche 
 * Informationen bereithält. Beispielsweiße benötigt ein Frame die Information,
 * wie viele Particle in diesem Frame drin sind. 
 * In diesem Beispiel wollen wir einen DeepContainer schreiben, welcher nur das 
 * n-te Element betrachtet. Als erstes betrachten wir hierfür den .
 * 
 * Zu betrachtende Beispiele
 * 1. 1d Fall (done
 * 2. 2d Fall
 * 3. nd Fall
 * 4. 1d Fall nur jedes zweites Element
 * 
 * 
 * 
 * 1. Fall
 * Arbeitsreihenfolge:
 * 1. Ich überlege mir, wie ich den DeepContainer auf den std::vector abbilde
 * 
 * Bedingungen:
 * Damit der Deepcontainer mit DeepForeach arbeitet benötigt er
 * 1. begin() und end()
 * 2. einen definierten iterator datentype (innerhalb der Klasse)
 * 
 * 2. Fall
 * 
 * @tparam TContainer ist der Container Der Daten
 * @tparam TElement Der Rückgabetyp des Iterator. in unserem ersten Beispiel ist
 * es ein int
 */
template<
    typename TContainer,
    typename TElement>
struct DeepContainer
{
public:
    typedef DeepIterator<TElement> iterator;
    
public:
    DeepContainer(TContainer& container):
        refContainer(container)
    {}
    
    iterator begin() {
        return DeepIterator<TElement>(*(refContainer.begin()));
    }
    
    
    iterator end() {
        return DeepIterator<TElement>(refContainer.size());
    }
    
    
    
protected:
    TContainer& refContainer;
};

/**
 * @brief Der DeepIterator ist ein Iterator über den DeepContainer. 
 * 
 */
template<
    typename TElement>
struct DeepIterator
{
public:
    typedef TElement        value_type;
    typedef TElement&       reference;
    typedef const reference const_reference;
    typedef TElement*       pointer;
    
public:
    
    DeepIterator(const size_t pos):
        pos(pos)
    {}
    
    DeepIterator(value_type& value):
        pos(0), ptr(&value)
    {}
    
    DeepIterator(value_type&& value):
        pos(0), ptr(&value)
    {}
    
    const_reference
    operator*()
    const
    {
        return ptr[pos];
    }
    
    reference
    operator*()
    {
        return ptr[pos];
    }
    
    bool
    operator!=(const DeepIterator& other)
    {
        return pos != other.pos;
    }
    
    void
    operator++()
    {
        pos++;
    }
    
    
protected:
    size_t pos;
    TElement* ptr;
};


int main(int argc, char **argv) {
    std::vector<int> vec1d = {0, 1, 2, 3, 4, 5};
    DeepContainer<std::vector<int>, int> deepCon1d(vec1d);
    
    deepForeach(deepCon1d, [](int& x){std::cout << x << " ";});

    SuperCell<Frame<Particle<int, 2>, 10> > cell(5, 2);
 
    std::cout << cell << std::endl;
    return EXIT_SUCCESS;
    
}
