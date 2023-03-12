# javascript

These hooks handle inputs, changehandling and database operations  
for note app. App can add people to notebook. These services use  
effect hooks and useState.

```javascript
import { useState, useEffect } from "react";
import axios from "axios";

export const useResource = (baseUrl) => {
  const [resources, setResources] = useState([]);
  const url = baseUrl;

  useEffect(() => {
    axios.get(url).then((initialResources) => {
      setResources(initialResources.data);
    });
  }, [url]);

  const create = async (resource) => {
    const response = await axios.post(url, resource);
    setResources(resources.concat(response.data));
    return response.data;
  };

  const service = {
    create,
  };

  return [resources, service];
};

export const useField = (type) => {
  const [value, setValue] = useState("");
  const onChange = (event) => {
    setValue(event.target.value);
  };
  return {
    type,
    value,
    onChange,
  };
};
```

This small snippet below uses the services above. App is based  
microservices. Code imports services and uses them to handle  
two diffent databases.

```javascript

import { useResource } from './hooks'
import { useField } from './hooks'

const App = () => {
  const content = useField('text')
  const name = useField('text')
  const number = useField('text')

  const [notes, noteService] = useResource('http://localhost:3005/notes')
  const [persons, personService] = useResource('http://localhost:3005/persons')

  const handleNoteSubmit = (event) => {
    event.preventDefault()
    noteService.create({ content: content.value })
  }

  const handlePersonSubmit = (event) => {
    event.preventDefault()
    personService.create({ name: name.value, number: number.value})
  }
```

[home](https://kaimhall.github.io/portfolio/)
