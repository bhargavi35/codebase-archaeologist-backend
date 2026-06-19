import React from 'react';
import { CustomInput } from './CustomInput'; // This import triggers the dependency link!

export const FormPage = () => {
  return <Form><CustomInput error="Name is required" /></Form>;
};